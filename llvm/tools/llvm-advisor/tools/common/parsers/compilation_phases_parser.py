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


class CompilationPhasesParser(BaseParser):
    def __init__(self):
        super().__init__(FileType.COMPILATION_PHASES)
        # Pattern for -ccc-print-bindings output: # "target" - "tool", inputs: [...], output: "..."
        self.binding_pattern = re.compile(
            r'^#\s+"([^"]+)"\s+-\s+"([^"]+)",\s+inputs:\s+\[([^\]]*)\],\s+output:\s+"([^"]*)"'
        )
        # Fallback patterns for other compilation phase formats
        self.phase_pattern = re.compile(r"^(\w+):\s*(.+)")
        self.timing_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|us)")

    def parse(self, file_path: str) -> ParsedFile:
        content = self.read_file_safe(file_path)
        if content is None:
            return self.create_parsed_file(
                file_path, [], {"error": "File too large or unreadable"}
            )

        try:
            lines = content.split("\n")
            phases_data = self._parse_compilation_phases(lines)

            total_time = sum(
                phase.get("duration", 0)
                for phase in phases_data["phases"]
                if phase.get("duration") is not None
            )

            metadata = {
                "file_size": self.get_file_size(file_path),
                "total_phases": len(phases_data["phases"]),
                "total_bindings": len(phases_data["bindings"]),
                "unique_tools": len(phases_data["tool_counts"]),
                "total_time": total_time,
                "time_unit": phases_data["time_unit"],
                "tool_counts": phases_data["tool_counts"],
            }

            return self.create_parsed_file(file_path, phases_data, metadata)

        except Exception as e:
            return self.create_parsed_file(file_path, [], {"error": str(e)})

    def _parse_compilation_phases(self, lines: List[str]) -> Dict[str, Any]:
        phases_data = {
            "phases": [],
            "bindings": [],
            "tool_counts": {},
            "time_unit": "ms",
            "summary": {},
            "clang_version": None,
            "target": None,
            "thread_model": None,
            "installed_dir": None,
            "file_type": None,  # Track what type of file this is
        }

        # First pass: determine file type based on content
        has_bindings = any(
            self.binding_pattern.match(line.strip()) for line in lines if line.strip()
        )
        has_compilation_phases_header = any(
            line.strip() == "COMPILATION PHASES:" for line in lines
        )

        if has_bindings:
            phases_data["file_type"] = "bindings"
        elif has_compilation_phases_header:
            phases_data["file_type"] = "phases"
        else:
            phases_data["file_type"] = "unknown"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse -ccc-print-bindings output (only for bindings files)
            if phases_data["file_type"] == "bindings":
                binding_match = self.binding_pattern.match(line)
                if binding_match:
                    target = binding_match.group(1)
                    tool = binding_match.group(2)
                    inputs_str = binding_match.group(3)
                    output = binding_match.group(4)

                    # Parse inputs array
                    inputs = []
                    if inputs_str.strip():
                        # Simple parsing of quoted inputs: "file1", "file2", ...
                        import re

                        input_matches = re.findall(r'"([^"]*)"', inputs_str)
                        inputs = input_matches

                    binding_entry = {
                        "target": target,
                        "tool": tool,
                        "inputs": inputs,
                        "output": output,
                    }

                    phases_data["bindings"].append(binding_entry)

                    # Count tools for summary
                    if tool in phases_data["tool_counts"]:
                        phases_data["tool_counts"][tool] += 1
                    else:
                        phases_data["tool_counts"][tool] = 1

                    continue

            # Extract compiler information (only for phases files)
            if phases_data["file_type"] == "phases":
                # Extract clang version
                if line.startswith("clang version"):
                    phases_data["clang_version"] = line
                    continue

                # Extract target
                if line.startswith("Target:"):
                    phases_data["target"] = line.replace("Target:", "").strip()
                    continue

                # Extract thread model
                if line.startswith("Thread model:"):
                    phases_data["thread_model"] = line.replace(
                        "Thread model:", ""
                    ).strip()
                    continue

                # Extract installed directory
                if line.startswith("InstalledDir:"):
                    phases_data["installed_dir"] = line.replace(
                        "InstalledDir:", ""
                    ).strip()
                    continue

            # Parse phase information (fallback for timing data)
            phase_match = self.phase_pattern.match(line)
            if phase_match:
                phase_name = phase_match.group(1)
                phase_info = phase_match.group(2)

                # Extract timing information if present
                timing_match = self.timing_pattern.search(phase_info)
                duration = None
                time_unit = "ms"

                if timing_match:
                    duration = float(timing_match.group(1))
                    time_unit = timing_match.group(2)

                    # Convert to consistent unit (milliseconds)
                    if time_unit == "s":
                        duration *= 1000
                    elif time_unit == "us":
                        duration /= 1000

                phase_entry = {
                    "name": phase_name,
                    "info": phase_info,
                    "duration": duration,
                    "time_unit": time_unit,
                }

                phases_data["phases"].append(phase_entry)
                continue

            # Handle simple timing lines like "Frontend: 123.45ms"
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    phase_name = parts[0].strip()
                    timing_info = parts[1].strip()

                    timing_match = self.timing_pattern.search(timing_info)
                    if timing_match:
                        duration = float(timing_match.group(1))
                        time_unit = timing_match.group(2)

                        # Convert to milliseconds
                        if time_unit == "s":
                            duration *= 1000
                        elif time_unit == "us":
                            duration /= 1000

                        phase_entry = {
                            "name": phase_name,
                            "info": timing_info,
                            "duration": duration,
                            "time_unit": "ms",
                        }

                        phases_data["phases"].append(phase_entry)

        # Calculate summary statistics
        durations = [
            p["duration"] for p in phases_data["phases"] if p["duration"] is not None
        ]
        phases_data["summary"] = {
            "total_time": sum(durations) if durations else 0,
            "avg_time": sum(durations) / len(durations) if durations else 0,
            "max_time": max(durations) if durations else 0,
            "min_time": min(durations) if durations else 0,
            "total_bindings": len(phases_data["bindings"]),
            "unique_tools": len(phases_data["tool_counts"]),
            "tool_counts": phases_data["tool_counts"],
        }

        return phases_data
