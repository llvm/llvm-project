# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import json
from typing import Dict, Any
from .base_parser import BaseParser
from ..models import FileType, ParsedFile


class ASTParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.AST_JSON)

    def parse(self, file_path: str) -> ParsedFile:
        if self.is_large_file(file_path):
            return self._parse_large_ast(file_path)

        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, {}, {"error": "File too large or unreadable"}
            )

        try:
            ast_data = json.loads(content)

            # Extract summary information
            summary = self._extract_ast_summary(ast_data)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "ast_summary": summary,
            }

            return self.create_parsed_file(file_path, ast_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _parse_large_ast(self, file_path: str) -> ParsedFile:
        try:
            # For large AST files, just extract basic info
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                # Read first chunk to get basic structure
                chunk = f.read(10000)  # 10KB

                # Try to parse at least the root node
                if chunk.startswith("{"):
                    bracket_count = 0
                    for i, char in enumerate(chunk):
                        if char == "{":
                            bracket_count += 1
                        elif char == "}":
                            bracket_count -= 1
                            if bracket_count == 0:
                                try:
                                    partial_data = json.loads(chunk[: i + 1])
                                    summary = self._extract_ast_summary(
                                        partial_data, partial=True
                                    )

                                    metadata = {
                                        "file_size": self.get_file_size(file_path),
                                        "ast_summary": summary,
                                        "is_partial": True,
                                    }

                                    return self.create_parsed_file(
                                        file_path, partial_data, metadata
                                    )
                                except:
                                    break

            metadata = {
                "file_size": self.get_file_size(file_path),
                "error": "File too large to parse completely",
            }

            return self.create_parsed_file(file_path, {}, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, {}, {"error": str(e)})

    def _extract_ast_summary(
        self, ast_data: Dict[str, Any], partial: bool = False
    ) -> Dict[str, Any]:
        summary = {
            "root_kind": ast_data.get("kind", "unknown"),
            "root_id": ast_data.get("id", "unknown"),
            "has_inner": "inner" in ast_data,
            "is_partial": partial,
        }

        if "inner" in ast_data and isinstance(ast_data["inner"], list):
            summary["inner_count"] = len(ast_data["inner"])

            # Count node types
            node_types = {}
            for node in ast_data["inner"]:
                if isinstance(node, dict) and "kind" in node:
                    kind = node["kind"]
                    node_types[kind] = node_types.get(kind, 0) + 1

            summary["node_types"] = node_types

        return summary
