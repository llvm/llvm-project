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
from ..models import FileType, ParsedFile


class DebugParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.DEBUG)
        self.dwarf_pattern = re.compile(r"^\s*<(\d+)><([0-9a-fA-F]+)>:\s*(.+)")
        self.debug_line_pattern = re.compile(
            r"^\s*Line\s+(\d+),\s*column\s+(\d+),\s*(.+)"
        )

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, {}, {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            debug_data = self._parse_debug_info(lines)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_lines": len(lines),
                **debug_data["summary"],
            }

            return self.create_parsed_file(file_path, debug_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _parse_debug_info(self, lines: List[str]) -> Dict[str, Any]:
        debug_data = {
            "dwarf_entries": [],
            "line_info": [],
            "sections": {},
            "summary": {},
        }

        current_section = None

        for line in lines:
            original_line = line
            line = line.strip()

            if not line:
                continue

            # Detect debug sections
            if line.startswith(".debug_"):
                current_section = line
                debug_data["sections"][current_section] = []
                continue

            # Parse DWARF entries
            dwarf_match = self.dwarf_pattern.match(original_line)
            if dwarf_match:
                entry = {
                    "depth": int(dwarf_match.group(1)),
                    "offset": dwarf_match.group(2),
                    "content": dwarf_match.group(3),
                }
                debug_data["dwarf_entries"].append(entry)

                if current_section:
                    debug_data["sections"][current_section].append(entry)
                continue

            # Parse debug line information
            line_match = self.debug_line_pattern.match(original_line)
            if line_match:
                line_info = {
                    "line": int(line_match.group(1)),
                    "column": int(line_match.group(2)),
                    "info": line_match.group(3),
                }
                debug_data["line_info"].append(line_info)

                if current_section:
                    debug_data["sections"][current_section].append(line_info)
                continue

            # Store other debug information by section
            if current_section:
                debug_data["sections"][current_section].append({"raw": line})

        debug_data["summary"] = {
            "dwarf_entry_count": len(debug_data["dwarf_entries"]),
            "line_info_count": len(debug_data["line_info"]),
            "section_count": len(debug_data["sections"]),
        }

        return debug_data
