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


class SymbolsParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.SYMBOLS)
        # Pattern for nm output: "address type name"
        self.nm_pattern = re.compile(r"^([0-9a-fA-F]+)\s+([A-Za-z?])\s+(.+)$")
        # Pattern for objdump symbol table
        self.objdump_pattern = re.compile(
            r"^([0-9a-fA-F]+)\s+([lgw!])\s+([dDfFoO])\s+(\S+)\s+([0-9a-fA-F]+)\s+(.+)$"
        )

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            symbols = self._parse_symbols(lines)

            # Calculate statistics
            symbol_types = {}
            sections = set()

            for symbol in symbols:
                if symbol.type:
                    symbol_types[symbol.type] = symbol_types.get(symbol.type, 0) + 1
                if symbol.section:
                    sections.add(symbol.section)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_symbols": len(symbols),
                "symbol_types": symbol_types,
                "unique_sections": len(sections),
            }

            return self.create_parsed_file(file_path, symbols, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_symbols(self, lines: List[str]) -> List[Symbol]:
        symbols = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Try nm format first
            nm_match = self.nm_pattern.match(line)
            if nm_match:
                symbol = Symbol(
                    name=nm_match.group(3),
                    address=nm_match.group(1),
                    type=nm_match.group(2),
                )
                symbols.append(symbol)
                continue

            # Try objdump format
            objdump_match = self.objdump_pattern.match(line)
            if objdump_match:
                symbol = Symbol(
                    name=objdump_match.group(6),
                    address=objdump_match.group(1),
                    type=objdump_match.group(3),
                    section=objdump_match.group(4),
                    size=(
                        int(objdump_match.group(5), 16)
                        if objdump_match.group(5) != "0"
                        else None
                    ),
                )
                symbols.append(symbol)
                continue

            # Try simple format: "name address size"
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                address = (
                    parts[1]
                    if parts[1].replace("0x", "").replace("0X", "").isalnum()
                    else None
                )
                size = None

                if len(parts) >= 3 and parts[2].isdigit():
                    size = int(parts[2])

                symbol = Symbol(name=name, address=address, size=size)
                symbols.append(symbol)

        return symbols
