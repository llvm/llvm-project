# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import re
from typing import Dict, List, Any
from .base_parser import BaseParser
from ..models import FileType, ParsedFile, BinarySize


class BinarySizeParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.BINARY_SIZE)
        # Pattern for size output like: "1234 5678 90 12345 section_name"
        self.size_pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+)$")
        # Pattern for nm-style output with size
        self.nm_pattern = re.compile(
            r"^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([A-Za-z])\s+(.+)$"
        )

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            size_data = self._parse_size_output(lines)

            total_size = sum(item.size for item in size_data)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_sections": len(size_data),
                "total_binary_size": total_size,
            }

            return self.create_parsed_file(file_path, size_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_size_output(self, lines: List[str]) -> List[BinarySize]:
        size_data = []
        total_size = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try standard size format first
            size_match = self.size_pattern.match(line)
            if size_match:
                text_size = int(size_match.group(1))
                data_size = int(size_match.group(2))
                bss_size = int(size_match.group(3))
                total = int(size_match.group(4))
                name = size_match.group(5)

                # Add individual sections if they have non-zero sizes
                if text_size > 0:
                    size_data.append(BinarySize(section=f"{name}.text", size=text_size))
                if data_size > 0:
                    size_data.append(BinarySize(section=f"{name}.data", size=data_size))
                if bss_size > 0:
                    size_data.append(BinarySize(section=f"{name}.bss", size=bss_size))

                total_size += total
                continue

            # Try nm-style format
            nm_match = self.nm_pattern.match(line)
            if nm_match:
                address = nm_match.group(1)
                size = int(nm_match.group(2), 16)
                symbol_type = nm_match.group(3)
                name = nm_match.group(4)

                size_data.append(BinarySize(section=name, size=size))
                total_size += size
                continue

            # Try to parse generic "section: size" format
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    section_name = parts[0].strip()
                    size_part = parts[1].strip()

                    # Extract numeric size
                    size_numbers = re.findall(r"\d+", size_part)
                    if size_numbers:
                        size = int(size_numbers[0])
                        size_data.append(BinarySize(section=section_name, size=size))
                        total_size += size

        # Calculate percentages
        if total_size > 0:
            for item in size_data:
                item.percentage = (item.size / total_size) * 100

        return size_data
