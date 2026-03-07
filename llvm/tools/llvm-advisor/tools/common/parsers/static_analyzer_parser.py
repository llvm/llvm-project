# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import re
from typing import List, Dict, Any
from .base_parser import BaseParser
from ..models import FileType, ParsedFile, Diagnostic, SourceLocation


class StaticAnalyzerParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.STATIC_ANALYZER)
        # Pattern for static analyzer output
        self.analyzer_pattern = re.compile(
            r"(?P<file>[^:]+):(?P<line>\d+):(?P<column>\d+):\s*(?P<level>\w+):\s*(?P<message>.+)"
        )

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            results = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Try to parse as diagnostic
                diagnostic = self._parse_analyzer_line(line)
                if diagnostic:
                    results.append(diagnostic)
                else:
                    # Store as raw line for other analysis results
                    results.append({"type": "raw", "content": line})

            # Count diagnostic types
            diagnostic_count = sum(1 for r in results if isinstance(r, Diagnostic))
            raw_count = len(results) - diagnostic_count

            metadata = {
                "total_results": len(results),
                "diagnostic_count": diagnostic_count,
                "raw_count": raw_count,
                "file_size": self.get_file_size(file_path),
            }

            return self.create_parsed_file(file_path, results, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_analyzer_line(self, line: str) -> Diagnostic:
        match = self.analyzer_pattern.match(line)
        if match:
            try:
                location = SourceLocation(
                    file=match.group("file"),
                    line=int(match.group("line")),
                    column=int(match.group("column")),
                )

                return Diagnostic(
                    level=match.group("level"),
                    message=match.group("message"),
                    location=location,
                )
            except ValueError:
                pass

        return None
