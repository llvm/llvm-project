# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from typing import Dict, Any
from .base_parser import BaseParser
from ..models import FileType, ParsedFile


class XRayParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.XRAY)

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            xray_data = {"entries": [], "functions": set(), "events": []}

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse XRay log entries
                parts = line.split()
                if len(parts) >= 4:
                    entry = {
                        "timestamp": (
                            parts[0] if parts[0].replace(".", "").isdigit() else None
                        ),
                        "function": parts[1] if len(parts) > 1 else None,
                        "event_type": parts[2] if len(parts) > 2 else None,
                        "data": " ".join(parts[3:]) if len(parts) > 3 else None,
                    }

                    xray_data["entries"].append(entry)

                    if entry["function"]:
                        xray_data["functions"].add(entry["function"])

                    if entry["event_type"]:
                        xray_data["events"].append(entry["event_type"])

            # Convert sets to lists for JSON serialization
            xray_data["functions"] = list(xray_data["functions"])

            metadata = {
                "total_entries": len(xray_data["entries"]),
                "unique_functions": len(xray_data["functions"]),
                "event_types": list(set(xray_data["events"])),
                "file_size": self.get_file_size(file_path),
            }

            return self.create_parsed_file(file_path, xray_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})
