# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import yaml
from typing import List, Dict, Any
from .base_parser import BaseParser
from ..models import FileType, ParsedFile, Remark, SourceLocation


class RemarksParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.REMARKS)

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            remarks = []
            # Handle custom YAML tags by creating a loader
            loader = yaml.SafeLoader
            loader.add_constructor(
                "!Passed", lambda loader, node: loader.construct_mapping(node)
            )
            loader.add_constructor(
                "!Missed", lambda loader, node: loader.construct_mapping(node)
            )
            loader.add_constructor(
                "!Analysis", lambda loader, node: loader.construct_mapping(node)
            )

            yaml_docs = yaml.load_all(content, Loader=loader)

            for doc in yaml_docs:
                if not doc:
                    continue

                remark = self._parse_remark(doc)
                if remark:
                    remarks.append(remark)

            metadata = {
                "total_remarks": len(remarks),
                "file_size": self.get_file_size(file_path),
            }

            return self.create_parsed_file(file_path, remarks, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_remark(self, doc: Dict[str, Any]) -> Remark:
        try:
            pass_name = doc.get("Pass", "")
            function = doc.get("Function", "")

            # Extract location information
            location = None
            debug_loc = doc.get("DebugLoc")
            if debug_loc:
                location = SourceLocation(
                    file=debug_loc.get("File"),
                    line=debug_loc.get("Line"),
                    column=debug_loc.get("Column"),
                )

            # Build message from args or use Name
            message = doc.get("Name", "")
            args = doc.get("Args", [])
            if args:
                arg_strings = []
                for arg in args:
                    if isinstance(arg, dict) and "String" in arg:
                        arg_strings.append(arg["String"])
                    elif isinstance(arg, str):
                        arg_strings.append(arg)
                if arg_strings:
                    message = "".join(arg_strings)

            return Remark(
                pass_name=pass_name,
                function=function,
                message=message,
                location=location,
                args=doc.get("Args", {}),
            )
        except Exception:
            return None
