# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import re
from typing import List
from .base_parser import BaseParser
from ..models import FileType, ParsedFile, Diagnostic, SourceLocation


class DiagnosticsParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.DIAGNOSTICS)
        # Pattern to match diagnostic lines like: "file.c:5:9: warning: message"
        self.diagnostic_pattern = re.compile(
            r"(?P<file>[^:]+):(?P<line>\d+):(?P<column>\d+):\s*(?P<level>\w+):\s*(?P<message>.+)"
        )

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            diagnostics = []
            lines = content.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                diagnostic = self._parse_diagnostic_line(line)
                if diagnostic:
                    diagnostics.append(diagnostic)

            # Count by level
            level_counts = {}
            for diag in diagnostics:
                level_counts[diag.level] = level_counts.get(diag.level, 0) + 1

            metadata = {
                "total_diagnostics": len(diagnostics),
                "level_counts": level_counts,
                "file_size": self.get_file_size(file_path),
            }

            return self.create_parsed_file(file_path, diagnostics, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_diagnostic_line(self, line: str) -> Diagnostic:
        match = self.diagnostic_pattern.match(line)
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

        # Fallback for lines that don't match the pattern
        if any(level in line.lower() for level in ["error", "warning", "note", "info"]):
            for level in ["error", "warning", "note", "info"]:
                if level in line.lower():
                    return Diagnostic(level=level, message=line)

        return None
