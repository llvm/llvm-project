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


class FTimeReportParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.FTIME_REPORT)
        # Patterns to match ftime-report output
        # Pattern for: 0.0112 (100.0%)   0.0020 (100.0%)   0.0132 (100.0%)   0.0132 (100.0%)  Front end
        # This needs to be reviewed with more files and outputs
        self.timing_line_pattern = re.compile(
            r"^\s*(\d+\.\d+)\s+\((\d+\.\d+)%\)\s+(\d+\.\d+)\s+\((\d+\.\d+)%\)\s+(\d+\.\d+)\s+\((\d+\.\d+)%\)\s+(\d+\.\d+)\s+\((\d+\.\d+)%\)\s+(.+)$"
        )
        self.total_pattern = re.compile(r"Total Execution Time:\s+(\d+\.\d+)\s+seconds")

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            timing_data = self._parse_ftime_report(lines)

            # Calculate statistics
            total_time = timing_data.get("total_execution_time", 0)
            timing_entries = timing_data.get("timings", [])

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_execution_time": total_time,
                "timing_entries_count": len(timing_entries),
                "top_time_consumer": (
                    timing_entries[0]["name"] if timing_entries else None
                ),
                "top_time_percentage": (
                    timing_entries[0]["percentage"] if timing_entries else 0
                ),
            }

            return self.create_parsed_file(file_path, timing_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_ftime_report(self, lines: List[str]) -> Dict[str, Any]:
        timing_data = {"timings": [], "total_execution_time": 0, "summary": {}}

        parsing_timings = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for total execution time
            total_match = self.total_pattern.search(line)
            if total_match:
                timing_data["total_execution_time"] = float(total_match.group(1))
                continue

            # Check if we're in the timing section
            if "---User Time---" in line and "--System Time--" in line:
                parsing_timings = True
                continue

            # Parse timing lines
            if parsing_timings:
                # Check if this line ends the timing section
                if not line or "===" in line:
                    parsing_timings = False
                    continue

                timing_match = self.timing_line_pattern.match(line)
                if timing_match:
                    user_time = float(timing_match.group(1))
                    user_percent = float(timing_match.group(2))
                    system_time = float(timing_match.group(3))
                    system_percent = float(timing_match.group(4))
                    total_time = float(timing_match.group(5))
                    total_percent = float(timing_match.group(6))
                    wall_time = float(timing_match.group(7))
                    wall_percent = float(timing_match.group(8))
                    name = timing_match.group(9).strip()

                    timing_entry = {
                        "name": name,
                        "user_time": user_time,
                        "user_percentage": user_percent,
                        "system_time": system_time,
                        "system_percentage": system_percent,
                        "total_time": total_time,
                        "total_percentage": total_percent,
                        "wall_time": wall_time,
                        "wall_percentage": wall_percent,
                        "time_seconds": wall_time,  # Use wall time as primary metric
                        "percentage": wall_percent,  # Use wall percentage as primary metric
                        "time_ms": wall_time * 1000,
                    }

                    timing_data["timings"].append(timing_entry)

        # Sort timings by time (descending)
        timing_data["timings"].sort(key=lambda x: x["time_seconds"], reverse=True)

        # Calculate summary
        if timing_data["timings"]:
            timing_data["summary"] = {
                "total_phases": len(timing_data["timings"]),
                "slowest_phase": timing_data["timings"][0]["name"],
                "slowest_time": timing_data["timings"][0]["time_seconds"],
                "fastest_phase": timing_data["timings"][-1]["name"],
                "fastest_time": timing_data["timings"][-1]["time_seconds"],
            }

        return timing_data
