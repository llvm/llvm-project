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


class AssemblyParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.ASSEMBLY)
        self.label_pattern = re.compile(r"^(\w+):")
        self.instruction_pattern = re.compile(r"^\s+(\w+)")
        self.section_pattern = re.compile(r"^\s*\.(text|data|bss|rodata)")

    def parse(self, file_path: str) -> ParsedFile:
        if self.is_large_file(file_path):
            return self._parse_large_assembly(file_path)

        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, {}, {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            asm_data = self._analyze_assembly_content(lines)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_lines": len(lines),
                **asm_data["summary"],
            }

            return self.create_parsed_file(file_path, asm_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _parse_large_assembly(self, file_path: str) -> ParsedFile:
        try:
            asm_data = {"labels": [], "instructions": {}, "sections": [], "summary": {}}
            line_count = 0

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_count += 1

                    # Only parse first 5000 lines for large files
                    if line_count > 5000:
                        break

                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith(";"):
                        continue

                    # Parse labels
                    label_match = self.label_pattern.match(line)
                    if label_match:
                        asm_data["labels"].append(label_match.group(1))

                    # Parse instructions
                    inst_match = self.instruction_pattern.match(line)
                    if inst_match:
                        inst = inst_match.group(1)
                        asm_data["instructions"][inst] = (
                            asm_data["instructions"].get(inst, 0) + 1
                        )

                    # Parse sections
                    section_match = self.section_pattern.match(line)
                    if section_match:
                        asm_data["sections"].append(section_match.group(1))

            asm_data["summary"] = {
                "label_count": len(asm_data["labels"]),
                "instruction_types": len(asm_data["instructions"]),
                "total_instructions": sum(asm_data["instructions"].values()),
                "section_count": len(set(asm_data["sections"])),
                "analyzed_lines": line_count,
                "is_partial": True,
            }

            metadata = {
                "file_size": self.get_file_size(file_path),
                **asm_data["summary"],
            }

            return self.create_parsed_file(file_path, asm_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _analyze_assembly_content(self, lines: List[str]) -> Dict[str, Any]:
        asm_data = {"labels": [], "instructions": {}, "sections": [], "summary": {}}

        for line in lines:
            original_line = line
            line = line.strip()

            if not line or line.startswith("#") or line.startswith(";"):
                continue

            # Parse labels
            label_match = self.label_pattern.match(line)
            if label_match:
                asm_data["labels"].append(label_match.group(1))
                continue

            # Parse instructions
            inst_match = self.instruction_pattern.match(original_line)
            if inst_match:
                inst = inst_match.group(1)
                asm_data["instructions"][inst] = (
                    asm_data["instructions"].get(inst, 0) + 1
                )
                continue

            # Parse sections
            section_match = self.section_pattern.match(line)
            if section_match:
                asm_data["sections"].append(section_match.group(1))

        asm_data["summary"] = {
            "label_count": len(asm_data["labels"]),
            "instruction_types": len(asm_data["instructions"]),
            "total_instructions": sum(asm_data["instructions"].values()),
            "section_count": len(set(asm_data["sections"])),
        }

        return asm_data
