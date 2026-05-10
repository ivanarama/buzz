import sys
import os
import logging
from dataclasses import dataclass
from itertools import groupby, pairwise
from typing import List, cast

import numpy as np
import torch

from buzz.model_loader import GIGAAM_REPO_ID, GIGAAM_REPO_REVISION, GIGAAM_KENLM_REPO_ID, model_root_dir
from buzz.transcriber.transcriber import Segment, FileTranscriptionTask

SAMPLE_RATE = 16_000
GIGAAM_FREQ = 25


def _load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    import subprocess

    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", file_path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


@dataclass(frozen=True)
class _AudioSegment:
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def audio_slice(self, sr: int = SAMPLE_RATE) -> slice:
        return slice(int(self.start_time * sr), int(self.end_time * sr))


def _chunk_audio(
    length: float, segment_length: float = 30, segment_shift: float = 20,
) -> list[_AudioSegment]:
    if length <= segment_length:
        return [_AudioSegment(0, length)]

    segments: list[tuple[float, float]] = []
    for start in np.arange(0, length - segment_length, step=segment_shift):
        segments.append((float(start), float(start) + segment_length))

    last_start, last_end = segments[-1]
    if last_end < length and length > last_start + segment_shift:
        segments.append((last_start + segment_shift, length))

    return [_AudioSegment(s, e) for s, e in segments]


def _groupby_into_spans(iterable):
    for key, group_iter in groupby(enumerate(iterable), key=lambda x: x[1]):
        group = list(group_iter)
        yield key, group[0][0], group[-1][0] + 1


def _merge_ctc_log_probs_by_blank_sep(segments, log_probs, tick_size, blank_id):
    tick_spans = []
    for seg, lp in zip(segments, log_probs):
        start_ticks = round(seg.start_time / tick_size)
        tick_spans.append((start_ticks, start_ticks + len(lp)))

    overlap_sizes: list[int] = []
    deltas: list[int] = []

    for ((_s, end), cur_lp), ((nxt_s, _ne), nxt_lp) in pairwise(
        zip(tick_spans, log_probs)
    ):
        overlap = end - nxt_s
        overlap_sizes.append(overlap)

        if overlap <= 0:
            deltas.append(0)
            continue

        blank_both = (
            (cur_lp[-overlap:].argmax(axis=1) == blank_id)
            & (nxt_lp[:overlap].argmax(axis=1) == blank_id)
        )
        if not np.any(blank_both):
            deltas.append(overlap // 2)
        else:
            blanks = [
                (i1, i2)
                for val, i1, i2 in _groupby_into_spans(blank_both)
                if val
            ]
            bs, be = max(blanks, key=lambda x: x[1] - x[0])
            deltas.append((be - 1 + bs) // 2)

    parts = []
    for idx, lp in enumerate(log_probs):
        cut_left = deltas[idx - 1] if idx > 0 else 0
        if idx < len(log_probs) - 1:
            cut_right = overlap_sizes[idx] - deltas[idx]
            parts.append(lp[cut_left:-cut_right] if cut_right > 0 else lp[cut_left:])
        else:
            parts.append(lp[cut_left:])

    return np.concatenate(parts, axis=0)


def _format_as_segments(words: list[dict], pause_threshold: float = 0.5) -> List[Segment]:
    if not words:
        return []

    groups: list[list[dict]] = [[words[0]]]
    for i in range(1, len(words)):
        if words[i]["start"] - words[i - 1]["end"] > pause_threshold:
            groups.append([words[i]])
        else:
            groups[-1].append(words[i])

    segments = []
    for seg_words in groups:
        segments.append(Segment(
            start=int(seg_words[0]["start"] * 1000),
            end=int(seg_words[-1]["end"] * 1000),
            text=" ".join(w["text"] for w in seg_words),
            translation="",
        ))
    return segments


class GigaAMTranscriber:
    @staticmethod
    def transcribe(task: FileTranscriptionTask) -> List[Segment]:
        import huggingface_hub
        import pyctcdecode
        from transformers import AutoModel

        if sys.stderr:
            sys.stderr.write("0%\n")

        force_cpu = os.getenv("BUZZ_FORCE_CPU", "false")
        device = "cuda" if (torch.cuda.is_available() and force_cpu == "false") else "cpu"

        # Load GigaAM v3_ctc model via transformers (no gigaam pip package needed)
        model_wrapper = AutoModel.from_pretrained(
            GIGAAM_REPO_ID,
            revision=GIGAAM_REPO_REVISION,
            trust_remote_code=True,
            cache_dir=model_root_dir,
        )
        gigaam_model = model_wrapper.model

        # Test CUDA compatibility with a dummy forward pass
        if device == "cuda":
            try:
                gigaam_model.to(device).eval()
                dummy = torch.zeros(1, 16000, device=device)
                dummy_len = torch.tensor([16000], device=device)
                with torch.inference_mode():
                    gigaam_model.forward(dummy, dummy_len)
                del dummy, dummy_len
            except (RuntimeError, torch.AcceleratorError):
                logging.warning("GigaAM CUDA not supported on this device, falling back to CPU")
                device = "cpu"
                gigaam_model.to(device)

        gigaam_model.eval()

        if sys.stderr:
            sys.stderr.write("20%\n")

        # Build vocab from model config (character-level for v3_ctc)
        vocabulary = gigaam_model.decoding.tokenizer.vocab
        blank_id: int = gigaam_model.decoding.blank_id
        vocab = list(vocabulary)
        vocab.insert(blank_id, "")

        tick_size: float = 1.0 / GIGAAM_FREQ

        if sys.stderr:
            sys.stderr.write("30%\n")

        # Load KenLM language model
        kenlm_path = huggingface_hub.hf_hub_download(
            GIGAAM_KENLM_REPO_ID,
            "kenlm.bin",
            cache_dir=model_root_dir,
        )

        # Build CTC decoder with KenLM
        decoder = pyctcdecode.build_ctcdecoder(
            labels=list(vocab),
            kenlm_model_path=str(kenlm_path),
            alpha=0.5,
            beta=1.0,
        )

        if sys.stderr:
            sys.stderr.write("40%\n")

        # Load audio
        waveform = _load_audio(task.file_path)
        length = len(waveform) / SAMPLE_RATE

        if sys.stderr:
            sys.stderr.write("50%\n")

        # Segment long audio
        audio_segments = _chunk_audio(length, segment_length=30, segment_shift=20)

        from torch.nn.utils.rnn import pad_sequence

        # Process each segment
        log_probs_list = []
        for i, seg in enumerate(audio_segments):
            part = waveform[seg.audio_slice()]
            with torch.inference_mode():
                tensor = torch.tensor(part, dtype=torch.float32).to(device)
                lengths = torch.tensor([len(part)], device=device)
                padded = tensor.unsqueeze(0)
                encoded, encoded_len = gigaam_model.forward(padded, lengths)
                lp = gigaam_model.head(encoder_output=encoded)
                log_probs_list.append(lp[0][:encoded_len[0]].cpu().numpy())

            progress = 50 + int(40 * (i + 1) / len(audio_segments))
            if sys.stderr:
                sys.stderr.write(f"{progress}%\n")

        # Merge overlapping segments
        if len(log_probs_list) == 1:
            merged = log_probs_list[0]
        else:
            merged = _merge_ctc_log_probs_by_blank_sep(
                audio_segments, log_probs_list, tick_size, blank_id,
            )

        # Decode with beam search
        merged_clipped = merged.clip(np.log(1e-15), 0)

        text = decoder.decode(merged_clipped, beam_width=100)
        text = text.strip()

        if sys.stderr:
            sys.stderr.write("95%\n")

        # Build segments from decoded text
        segments = []
        if not text:
            if sys.stderr:
                sys.stderr.write("100%\n")
            return segments

        # Try to get word-level timestamps from beam results
        words = []
        try:
            beams = decoder.decode_beams(merged_clipped, beam_width=5)
            if beams:
                best_beam = beams[0]
                # Try text_frames: list of (word, (start_frame, end_frame))
                text_frames = getattr(best_beam, 'text_frames', None)
                if text_frames:
                    for word, (s, e) in text_frames:
                        if word.strip():
                            words.append({
                                "text": word,
                                "start": round(s * tick_size, 3),
                                "end": round(e * tick_size, 3),
                            })
                else:
                    # Try word_start_times / word_end_times
                    ws = getattr(best_beam, 'word_start_times', None)
                    we = getattr(best_beam, 'word_end_times', None)
                    wl = getattr(best_beam, 'words', None) or getattr(best_beam, '_words', None)
                    if wl and ws and we:
                        for i, word in enumerate(wl):
                            if word.strip() and i < len(ws) and i < len(we):
                                words.append({
                                    "text": word,
                                    "start": round(ws[i] * tick_size, 3),
                                    "end": round(we[i] * tick_size, 3),
                                })
        except Exception:
            logging.exception("Failed to get word timestamps from beam search")

        if words:
            segments = _format_as_segments(words)
        else:
            # Fallback: single segment with full text
            duration_ms = int(length * 1000)
            segments = [Segment(start=0, end=duration_ms, text=text, translation="")]

        if sys.stderr:
            sys.stderr.write("100%\n")

        return segments
