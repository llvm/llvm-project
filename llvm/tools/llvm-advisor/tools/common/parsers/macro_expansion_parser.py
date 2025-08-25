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


class MacroExpansionParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.MACRO_EXPANSION)
        self.macro_pattern = re.compile(r"^#define\s+(\w+)(?:\(([^)]*)\))?\s*(.*)")
        self.expansion_pattern = re.compile(
            r'^\s*//\s*expanded\s*from\s*[\'"]([^\'"]+)[\'"]'
        )

    def parse(self, file_path: str) -> ParsedFile:
        if self.is_large_file(file_path):
            return self._parse_large_macro_expansion(file_path)

        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, {}, {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            macro_data = self._analyze_macro_content(lines)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_lines": len(lines),
                **macro_data["summary"],
            }

            return self.create_parsed_file(file_path, macro_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _parse_large_macro_expansion(self, file_path: str) -> ParsedFile:
        try:
            macro_data = {"macros": {}, "expansions": [], "summary": {}}

            line_count = 0

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_count += 1

                    # Only analyze first 5000 lines for large files
                    if line_count > 5000:
                        break

                    line = line.strip()

                    if not line:
                        continue

                    # Parse macro definitions
                    macro_match = self.macro_pattern.match(line)
                    if macro_match:
                        macro_name = macro_match.group(1)
                        params = macro_match.group(2)
                        definition = macro_match.group(3)

                        macro_data["macros"][macro_name] = {
                            "parameters": params.split(",") if params else [],
                            "definition": definition,
                        }
                        continue

                    # Parse expansion comments
                    expansion_match = self.expansion_pattern.match(line)
                    if expansion_match:
                        macro_data["expansions"].append(expansion_match.group(1))

            macro_data["summary"] = {
                "macro_count": len(macro_data["macros"]),
                "expansion_count": len(macro_data["expansions"]),
                "analyzed_lines": line_count,
                "is_partial": True,
            }

            metadata = {
                "file_size": self.get_file_size(file_path),
                **macro_data["summary"],
            }

            return self.create_parsed_file(file_path, macro_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _analyze_macro_content(self, lines: List[str]) -> Dict[str, Any]:
        macro_data = {"macros": {}, "expansions": [], "summary": {}}

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Parse macro definitions
            macro_match = self.macro_pattern.match(line)
            if macro_match:
                macro_name = macro_match.group(1)
                params = macro_match.group(2)
                definition = macro_match.group(3)

                macro_data["macros"][macro_name] = {
                    "parameters": params.split(",") if params else [],
                    "definition": definition,
                }
                continue

            # Parse expansion comments
            expansion_match = self.expansion_pattern.match(line)
            if expansion_match:
                macro_data["expansions"].append(expansion_match.group(1))

        macro_data["summary"] = {
            "macro_count": len(macro_data["macros"]),
            "expansion_count": len(macro_data["expansions"]),
        }

        return macro_data
