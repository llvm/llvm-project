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


class IRParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.IR)
        self.function_pattern = re.compile(r"^define\s+.*@(\w+)\s*\(")
        self.global_pattern = re.compile(r"^@(\w+)\s*=")
        self.type_pattern = re.compile(r"^%(\w+)\s*=\s*type")

    def parse(self, file_path: str) -> ParsedFile:
        if self.is_large_file(file_path):
            return self._parse_large_ir(file_path)

        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, {}, {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            ir_data = self._analyze_ir_content(lines)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_lines": len(lines),
                **ir_data["summary"],
            }

            return self.create_parsed_file(file_path, ir_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _parse_large_ir(self, file_path: str) -> ParsedFile:
        try:
            ir_data = {"functions": [], "globals": [], "types": [], "summary": {}}
            line_count = 0

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_count += 1
                    line = line.strip()

                    # Only parse first 10000 lines for large files
                    if line_count > 10000:
                        break

                    if line.startswith("define"):
                        match = self.function_pattern.match(line)
                        if match:
                            ir_data["functions"].append(match.group(1))
                    elif line.startswith("@") and "=" in line:
                        match = self.global_pattern.match(line)
                        if match:
                            ir_data["globals"].append(match.group(1))
                    elif line.startswith("%") and "type" in line:
                        match = self.type_pattern.match(line)
                        if match:
                            ir_data["types"].append(match.group(1))

            ir_data["summary"] = {
                "function_count": len(ir_data["functions"]),
                "global_count": len(ir_data["globals"]),
                "type_count": len(ir_data["types"]),
                "analyzed_lines": line_count,
                "is_partial": True,
            }

            metadata = {
                "file_size": self.get_file_size(file_path),
                **ir_data["summary"],
            }

            return self.create_parsed_file(file_path, ir_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _analyze_ir_content(self, lines: List[str]) -> Dict[str, Any]:
        ir_data = {
            "functions": [],
            "globals": [],
            "types": [],
            "instructions": {},
            "summary": {},
        }

        instruction_count = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            # Count instructions
            if any(
                line.strip().startswith(inst)
                for inst in ["%", "call", "ret", "br", "load", "store"]
            ):
                instruction_count += 1

            # Extract functions
            if line.startswith("define"):
                match = self.function_pattern.match(line)
                if match:
                    ir_data["functions"].append(match.group(1))

            # Extract globals
            elif line.startswith("@") and "=" in line:
                match = self.global_pattern.match(line)
                if match:
                    ir_data["globals"].append(match.group(1))

            # Extract types
            elif line.startswith("%") and "type" in line:
                match = self.type_pattern.match(line)
                if match:
                    ir_data["types"].append(match.group(1))

        ir_data["summary"] = {
            "function_count": len(ir_data["functions"]),
            "global_count": len(ir_data["globals"]),
            "type_count": len(ir_data["types"]),
            "instruction_count": instruction_count,
        }

        return ir_data
