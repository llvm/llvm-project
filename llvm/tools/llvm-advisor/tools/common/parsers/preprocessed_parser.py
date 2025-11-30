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


class PreprocessedParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.PREPROCESSED)
        self.line_directive_pattern = re.compile(r'^#\s*(\d+)\s+"([^"]+)"')
        self.pragma_pattern = re.compile(r"^#\s*pragma\s+(.+)")

    def parse(self, file_path: str) -> ParsedFile:
        if self.is_large_file(file_path):
            return self._parse_large_preprocessed(file_path)

        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, {}, {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            preprocessed_data = self._analyze_preprocessed_content(lines)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_lines": len(lines),
                **preprocessed_data["summary"],
            }

            return self.create_parsed_file(file_path, preprocessed_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _parse_large_preprocessed(self, file_path: str) -> ParsedFile:
        try:
            preprocessed_data = {
                "source_files": set(),
                "pragmas": [],
                "directives": [],
                "summary": {},
            }

            line_count = 0
            code_lines = 0

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_count += 1

                    # Only analyze first 10000 lines for large files
                    if line_count > 10000:
                        break

                    line = line.strip()

                    if not line:
                        continue

                    # Parse line directives
                    line_match = self.line_directive_pattern.match(line)
                    if line_match:
                        source_file = line_match.group(2)
                        preprocessed_data["source_files"].add(source_file)
                        preprocessed_data["directives"].append(
                            {"line": int(line_match.group(1)), "file": source_file}
                        )
                        continue

                    # Parse pragma directives
                    pragma_match = self.pragma_pattern.match(line)
                    if pragma_match:
                        preprocessed_data["pragmas"].append(pragma_match.group(1))
                        continue

                    # Count actual code lines
                    if not line.startswith("#"):
                        code_lines += 1

            preprocessed_data["source_files"] = list(preprocessed_data["source_files"])

            preprocessed_data["summary"] = {
                "source_file_count": len(preprocessed_data["source_files"]),
                "pragma_count": len(preprocessed_data["pragmas"]),
                "directive_count": len(preprocessed_data["directives"]),
                "code_lines": code_lines,
                "analyzed_lines": line_count,
                "is_partial": True,
            }

            metadata = {
                "file_size": self.get_file_size(file_path),
                **preprocessed_data["summary"],
            }

            return self.create_parsed_file(file_path, preprocessed_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _analyze_preprocessed_content(self, lines: List[str]) -> Dict[str, Any]:
        preprocessed_data = {
            "source_files": set(),
            "pragmas": [],
            "directives": [],
            "summary": {},
        }

        code_lines = 0

        for line in lines:
            original_line = line
            line = line.strip()

            if not line:
                continue

            # Parse line directives
            line_match = self.line_directive_pattern.match(line)
            if line_match:
                source_file = line_match.group(2)
                preprocessed_data["source_files"].add(source_file)
                preprocessed_data["directives"].append(
                    {"line": int(line_match.group(1)), "file": source_file}
                )
                continue

            # Parse pragma directives
            pragma_match = self.pragma_pattern.match(line)
            if pragma_match:
                preprocessed_data["pragmas"].append(pragma_match.group(1))
                continue

            # Count actual code lines (not preprocessor directives)
            if not line.startswith("#"):
                code_lines += 1

        # Convert set to list for JSON serialization
        preprocessed_data["source_files"] = list(preprocessed_data["source_files"])

        preprocessed_data["summary"] = {
            "source_file_count": len(preprocessed_data["source_files"]),
            "pragma_count": len(preprocessed_data["pragmas"]),
            "directive_count": len(preprocessed_data["directives"]),
            "code_lines": code_lines,
        }

        return preprocessed_data
