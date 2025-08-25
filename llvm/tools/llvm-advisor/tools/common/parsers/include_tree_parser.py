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
from ..models import FileType, ParsedFile, Dependency


class IncludeTreeParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.INCLUDE_TREE)
        self.include_pattern = re.compile(r"^(\s*)(\S+)\s*(?:\(([^)]+)\))?")

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            include_data = self._parse_include_tree(lines)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_includes": len(include_data["dependencies"]),
                "unique_files": len(include_data["files"]),
                "max_depth": include_data["max_depth"],
            }

            return self.create_parsed_file(file_path, include_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_include_tree(self, lines: List[str]) -> Dict[str, Any]:
        include_data = {"dependencies": [], "files": set(), "tree": [], "max_depth": 0}

        stack = []  # Stack to track parent files

        for line in lines:
            if not line.strip():
                continue

            match = self.include_pattern.match(line)
            if match:
                indent = len(match.group(1))
                file_path = match.group(2)
                extra_info = match.group(3)

                depth = indent // 2  # Assuming 2 spaces per indent level
                include_data["max_depth"] = max(include_data["max_depth"], depth)

                # Adjust stack based on current depth
                while len(stack) > depth:
                    stack.pop()

                # Add to files set
                include_data["files"].add(file_path)

                # Create dependency relationship
                if stack:
                    parent = stack[-1]
                    dependency = Dependency(
                        source=parent, target=file_path, type="include"
                    )
                    include_data["dependencies"].append(dependency)

                # Add tree entry
                tree_entry = {
                    "file": file_path,
                    "depth": depth,
                    "parent": stack[-1] if stack else None,
                    "extra_info": extra_info,
                }
                include_data["tree"].append(tree_entry)

                # Push current file onto stack
                stack.append(file_path)

        # Convert set to list for JSON serialization
        include_data["files"] = list(include_data["files"])

        return include_data
