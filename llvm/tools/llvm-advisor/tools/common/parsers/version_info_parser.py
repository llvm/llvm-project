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


class VersionInfoParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.VERSION_INFO)
        self.version_pattern = re.compile(r"clang version\s+([\d\w\.\-\+]+)")
        self.target_pattern = re.compile(r"Target:\s+(.+)")
        self.thread_pattern = re.compile(r"Thread model:\s+(.+)")

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            version_data = self._parse_version_info(lines)

            metadata = {
                "file_size": self.get_file_size(file_path),
                "clang_version": version_data.get("clang_version"),
                "target": version_data.get("target"),
                "thread_model": version_data.get("thread_model"),
                "full_version_string": version_data.get("full_version_string"),
            }

            return self.create_parsed_file(file_path, version_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_version_info(self, lines: List[str]) -> Dict[str, Any]:
        version_data = {
            "clang_version": None,
            "target": None,
            "thread_model": None,
            "full_version_string": "",
            "install_dir": None,
            "libraries": [],
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Store the full version string (usually the first line)
            if not version_data["full_version_string"] and "clang version" in line:
                version_data["full_version_string"] = line

                # Extract version number
                version_match = self.version_pattern.search(line)
                if version_match:
                    version_data["clang_version"] = version_match.group(1)

            # Extract target
            target_match = self.target_pattern.search(line)
            if target_match:
                version_data["target"] = target_match.group(1)

            # Extract thread model
            thread_match = self.thread_pattern.search(line)
            if thread_match:
                version_data["thread_model"] = thread_match.group(1)

            # Extract install directory
            if line.startswith("InstalledDir:"):
                version_data["install_dir"] = line.replace("InstalledDir:", "").strip()

            # Extract library paths
            if "library" in line.lower() or "lib" in line:
                version_data["libraries"].append(line)

        return version_data
