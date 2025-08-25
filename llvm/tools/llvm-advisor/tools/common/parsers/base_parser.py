# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
from ..models import ParsedFile, FileType


class BaseParser(ABC):
    def __init__(self, file_type: FileType):
        self.file_type = file_type

    @abstractmethod
    def parse(self, file_path: str) -> ParsedFile:
        pass

    def can_parse(self, file_path: str) -> bool:
        return os.path.exists(file_path) and os.path.isfile(file_path)

    def get_file_size(self, file_path: str) -> int:
        return os.path.getsize(file_path) if os.path.exists(file_path) else 0

    def is_large_file(self, file_path: str, threshold: int = 100 * 1024 * 1024) -> bool:
        return self.get_file_size(file_path) > threshold

    def read_file_safe(
        self, file_path: str, max_size: int = 100 * 1024 * 1024
    ) -> Optional[str]:
        try:
            if self.is_large_file(file_path, max_size):
                return None
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return None

    def read_file_chunked(self, file_path: str, chunk_size: int = 1024 * 1024):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception:
            return

    def create_parsed_file(
        self, file_path: str, data: Any, metadata: Dict[str, Any] = None
    ) -> ParsedFile:
        return ParsedFile(
            file_type=self.file_type,
            file_path=file_path,
            data=data,
            metadata=metadata or {},
        )
