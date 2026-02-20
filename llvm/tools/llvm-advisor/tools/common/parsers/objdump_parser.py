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
from ..models import FileType, ParsedFile, Symbol


class ObjdumpParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.OBJDUMP)
        self.symbol_pattern = re.compile(
            r"^([0-9a-fA-F]+)\s+([lgw!])\s+([dDfFoO])\s+(\S+)\s+([0-9a-fA-F]+)\s+(.+)"
        )
        self.section_pattern = re.compile(
            r"^Idx\s+Name\s+Size\s+VMA\s+LMA\s+File Offset\s+Algn"
        )
        self.disasm_pattern = re.compile(
            r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F\s]+)\s+(.+)"
        )

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, {}, {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            objdump_data = self._parse_objdump_content(lines)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_lines": len(lines),
                **objdump_data["summary"],
            }

            return self.create_parsed_file(file_path, objdump_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _parse_objdump_content(self, lines: List[str]) -> Dict[str, Any]:
        objdump_data = {
            "symbols": [],
            "sections": [],
            "disassembly": [],
            "headers": [],
            "summary": {},
        }

        current_section = None
        in_symbol_table = False
        in_disassembly = False

        for line in lines:
            line = line.rstrip()

            if not line:
                continue

            # Detect sections
            if "SYMBOL TABLE:" in line:
                in_symbol_table = True
                continue
            elif "Disassembly of section" in line:
                in_disassembly = True
                current_section = line
                continue
            elif line.startswith("Contents of section"):
                current_section = line
                continue

            # Parse symbol table
            if in_symbol_table and self.symbol_pattern.match(line):
                match = self.symbol_pattern.match(line)
                if match:
                    symbol = Symbol(
                        name=match.group(6),
                        address=match.group(1),
                        type=match.group(3),
                        section=match.group(4),
                    )
                    objdump_data["symbols"].append(symbol)

            # Parse disassembly
            elif in_disassembly and self.disasm_pattern.match(line):
                match = self.disasm_pattern.match(line)
                if match:
                    objdump_data["disassembly"].append(
                        {
                            "address": match.group(1),
                            "bytes": match.group(2).strip(),
                            "instruction": match.group(3),
                        }
                    )

            # Collect headers and other info
            elif line.startswith("Program Header:") or line.startswith(
                "Section Headers:"
            ):
                objdump_data["headers"].append(line)

        objdump_data["summary"] = {
            "symbol_count": len(objdump_data["symbols"]),
            "disasm_count": len(objdump_data["disassembly"]),
            "section_count": len(objdump_data["sections"]),
            "header_count": len(objdump_data["headers"]),
        }

        return objdump_data
