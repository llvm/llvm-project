# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from typing import Dict, List, Any
from .base_parser import BaseParser
from ..models import FileType, ParsedFile


class PGOProfileParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.PGO_PROFILE)

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, {}, {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            profile_data = {"functions": [], "counters": [], "raw_lines": []}

            current_function = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                profile_data["raw_lines"].append(line)

                # Simple pattern matching for PGO profile data
                if line.startswith("# Func Hash:") or line.startswith("Function:"):
                    current_function = line
                    profile_data["functions"].append(line)
                elif line.startswith("# Num Counters:") or line.isdigit():
                    profile_data["counters"].append(line)

            metadata = {
                "total_functions": len(profile_data["functions"]),
                "total_counters": len(profile_data["counters"]),
                "total_lines": len(profile_data["raw_lines"]),
                "file_size": self.get_file_size(file_path),
            }

            return self.create_parsed_file(file_path, profile_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})
