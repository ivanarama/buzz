import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QObject

from buzz.transcriber.file_transcriber import FileTranscriber, app_env, is_video_file
from buzz.transcriber.transcriber import FileTranscriptionTask, Segment


class TOneFileTranscriber(FileTranscriber):
    """T-one transcriber for Russian telephony ASR."""

    def __init__(self, task: FileTranscriptionTask, parent: Optional["QObject"] = None):
        super().__init__(task, parent)
        self.pipeline = None

    def transcribe(self) -> List[Segment]:
        logging.debug(
            "Starting T-one file transcription, file path = %s",
            self.transcription_task.file_path,
        )

        try:
            from tone import StreamingCTCPipeline, read_audio
        except ImportError:
            raise Exception(
                "T-one library not found. Install with: pip install tone"
            )

        # Get the audio file path (extract from video if needed)
        file_path = self.transcription_task.file_path
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")

        # Extract audio from video if needed
        if is_video_file(file_path):
            logging.debug("Video file detected, extracting audio...")
            self.progress.emit((5, 100))
            file_path = self._extract_audio_from_video(file_path)
            self.progress.emit((10, 100))

        # Load audio file
        try:
            audio = read_audio(file_path)
            logging.debug("Audio loaded successfully")
            self.progress.emit((30, 100))
        except Exception as e:
            logging.error("Failed to read audio file: %s", e)
            raise Exception(f"Failed to read audio file: {e}")

        # Initialize T-one pipeline
        try:
            self.pipeline = StreamingCTCPipeline.from_hugging_face()
            logging.debug("T-one pipeline loaded successfully")
            self.progress.emit((50, 100))
        except Exception as e:
            logging.error("Failed to load T-one pipeline: %s", e)
            raise Exception(f"Failed to load T-one model: {e}")

        # Run offline transcription
        try:
            self.progress.emit((70, 100))
            result = self.pipeline.forward_offline(audio)
            logging.debug("T-one transcription completed, %d phrases", len(result))
            self.progress.emit((90, 100))
        except Exception as e:
            logging.error("T-one transcription failed: %s", e)
            raise Exception(f"Transcription failed: {e}")

        # Convert T-one output to Buzz Segment format
        segments = []
        for phrase in result:
            segments.append(Segment(
                start=int(phrase.start_time * 1000),  # Convert seconds to ms
                end=int(phrase.end_time * 1000),      # Convert seconds to ms
                text=phrase.text.strip()
            ))

        logging.debug("Converted %d segments to Buzz format", len(segments))
        self.progress.emit((100, 100))
        return segments

    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file using ffmpeg and return the audio file path."""
        # Create temporary file for extracted audio
        temp_audio = tempfile.mktemp(suffix=".wav")
        temp_audio = str(Path(temp_audio).resolve())

        # Use ffmpeg to extract audio
        cmd = [
            "ffmpeg",
            "-threads", "0",
            "-loglevel", "panic",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "8000",  # 8kHz sample rate for T-one
            "-ac", "1",  # Mono
            temp_audio
        ]

        if sys.platform == "win32":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            result = subprocess.run(
                cmd,
                capture_output=True,
                startupinfo=si,
                env=app_env,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
        else:
            result = subprocess.run(cmd, capture_output=True, env=app_env)

        if result.returncode != 0:
            if result.stderr:
                error_msg = result.stderr.decode()
                logging.error(f"FFmpeg error: {error_msg}")
            raise Exception(f"Failed to extract audio from video: ffmpeg returned code {result.returncode}")

        logging.debug(f"Audio extracted to: {temp_audio}")
        return temp_audio

    def stop(self) -> None:
        """Stop transcription and clean up resources."""
        logging.debug("T-one transcriber stop called")
        self.pipeline = None
