# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import os
import sys
from collections import defaultdict, Counter
from typing import Dict, Any
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
tools_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(tools_dir))

from common.models import FileType, Diagnostic
from ..base import BaseEndpoint, APIResponse


class DiagnosticsEndpoint(BaseEndpoint):
    """Specialized endpoints for compiler diagnostics analysis"""

    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        """Route requests to specific handlers based on path"""
        if len(path_parts) >= 3:
            sub_endpoint = path_parts[2]
        else:
            sub_endpoint = "overview"

        method_name = f"handle_{sub_endpoint.replace('-', '_')}"

        if hasattr(self, method_name):
            handler_method = getattr(self, method_name)
            return handler_method(path_parts, query_params)
        else:
            available_methods = [
                method[7:].replace("_", "-")
                for method in dir(self)
                if method.startswith("handle_") and method != "handle"
            ]

            return APIResponse.error(
                f"Sub-endpoint '{sub_endpoint}' not available. "
                f"Available: {', '.join(available_methods)}",
                404,
            )

    def handle_overview(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/diagnostics/overview - Overall diagnostics statistics"""
        parsed_data = self.get_parsed_data()

        level_counts = Counter()
        file_counts = Counter()
        total_diagnostics = 0

        for unit_name, unit_data in parsed_data.items():
            if FileType.DIAGNOSTICS in unit_data:
                for parsed_file in unit_data[FileType.DIAGNOSTICS]:
                    if isinstance(parsed_file.data, list):
                        total_diagnostics += len(parsed_file.data)

                        for diagnostic in parsed_file.data:
                            if isinstance(diagnostic, Diagnostic):
                                level_counts[diagnostic.level] += 1

                                if diagnostic.location and diagnostic.location.file:
                                    file_counts[diagnostic.location.file] += 1

        overview_data = {
            "totals": {
                "diagnostics": total_diagnostics,
                "files_with_issues": len(file_counts),
                "error_rate": level_counts.get("error", 0)
                / max(total_diagnostics, 1)
                * 100,
                "warning_rate": level_counts.get("warning", 0)
                / max(total_diagnostics, 1)
                * 100,
            },
            "by_level": dict(level_counts),
            "top_problematic_files": dict(file_counts.most_common(10)),
        }

        return APIResponse.success(overview_data)

    def handle_by_level(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/diagnostics/levels - Analysis by diagnostic levels (error, warning, note)"""
        parsed_data = self.get_parsed_data()

        levels_data = defaultdict(
            lambda: {"count": 0, "files": set(), "messages": Counter(), "examples": []}
        )

        for unit_name, unit_data in parsed_data.items():
            if FileType.DIAGNOSTICS in unit_data:
                for parsed_file in unit_data[FileType.DIAGNOSTICS]:
                    if isinstance(parsed_file.data, list):
                        for diagnostic in parsed_file.data:
                            if isinstance(diagnostic, Diagnostic):
                                level = diagnostic.level
                                levels_data[level]["count"] += 1

                                if diagnostic.location and diagnostic.location.file:
                                    levels_data[level]["files"].add(
                                        diagnostic.location.file
                                    )

                                # Count similar messages
                                message_key = diagnostic.message[
                                    :50
                                ]  # First 50 chars as key
                                levels_data[level]["messages"][message_key] += 1

                                # Keep examples
                                if len(levels_data[level]["examples"]) < 5:
                                    example = {
                                        "message": diagnostic.message,
                                        "location": {
                                            "file": (
                                                diagnostic.location.file
                                                if diagnostic.location
                                                else None
                                            ),
                                            "line": (
                                                diagnostic.location.line
                                                if diagnostic.location
                                                else None
                                            ),
                                            "column": (
                                                diagnostic.location.column
                                                if diagnostic.location
                                                else None
                                            ),
                                        },
                                        "code": diagnostic.code,
                                    }
                                    levels_data[level]["examples"].append(example)

        # Convert to serializable format
        result = {}
        for level, data in levels_data.items():
            result[level] = {
                "count": data["count"],
                "unique_files": len(data["files"]),
                "common_messages": dict(data["messages"].most_common(5)),
                "examples": data["examples"],
            }

        return APIResponse.success({"levels": result, "total_levels": len(result)})

    def handle_files(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/diagnostics/files - Analysis by files with issues"""
        parsed_data = self.get_parsed_data()

        files_data = defaultdict(
            lambda: {
                "diagnostics": [],
                "level_counts": Counter(),
                "lines_with_issues": set(),
            }
        )

        for unit_name, unit_data in parsed_data.items():
            if FileType.DIAGNOSTICS in unit_data:
                for parsed_file in unit_data[FileType.DIAGNOSTICS]:
                    if isinstance(parsed_file.data, list):
                        for diagnostic in parsed_file.data:
                            if (
                                isinstance(diagnostic, Diagnostic)
                                and diagnostic.location
                                and diagnostic.location.file
                            ):
                                file_path = diagnostic.location.file

                                files_data[file_path]["level_counts"][
                                    diagnostic.level
                                ] += 1

                                if diagnostic.location.line:
                                    files_data[file_path]["lines_with_issues"].add(
                                        diagnostic.location.line
                                    )

                                diagnostic_info = {
                                    "level": diagnostic.level,
                                    "message": diagnostic.message,
                                    "line": diagnostic.location.line,
                                    "column": diagnostic.location.column,
                                    "code": diagnostic.code,
                                }
                                files_data[file_path]["diagnostics"].append(
                                    diagnostic_info
                                )

        # Convert to serializable format
        result = {}
        for file_path, data in files_data.items():
            total_issues = sum(data["level_counts"].values())
            result[file_path] = {
                "file_name": os.path.basename(file_path),
                "total_diagnostics": total_issues,
                "level_breakdown": dict(data["level_counts"]),
                "lines_affected": len(data["lines_with_issues"]),
                "diagnostics": sorted(
                    data["diagnostics"],
                    key=lambda x: (x.get("line", 0), x.get("column", 0)),
                ),
            }

        # Sort by total diagnostics count
        sorted_files = dict(
            sorted(
                result.items(), key=lambda x: x[1]["total_diagnostics"], reverse=True
            )
        )

        return APIResponse.success(
            {"files": sorted_files, "total_files_with_issues": len(sorted_files)}
        )

    def handle_patterns(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/diagnostics/patterns - Common diagnostic patterns and trends"""
        parsed_data = self.get_parsed_data()

        message_patterns = Counter()
        code_patterns = Counter()
        line_distribution = defaultdict(list)

        for unit_name, unit_data in parsed_data.items():
            if FileType.DIAGNOSTICS in unit_data:
                for parsed_file in unit_data[FileType.DIAGNOSTICS]:
                    if isinstance(parsed_file.data, list):
                        for diagnostic in parsed_file.data:
                            if isinstance(diagnostic, Diagnostic):
                                # Pattern analysis on messages
                                words = diagnostic.message.lower().split()
                                if len(words) >= 2:
                                    pattern = " ".join(
                                        words[:3]
                                    )  # First 3 words as pattern
                                    message_patterns[pattern] += 1

                                # Code patterns
                                if diagnostic.code:
                                    code_patterns[diagnostic.code] += 1

                                # Line distribution for hotspot analysis
                                if (
                                    diagnostic.location
                                    and diagnostic.location.file
                                    and diagnostic.location.line
                                ):
                                    line_distribution[diagnostic.location.file].append(
                                        diagnostic.location.line
                                    )

        # Find line clusters (areas with many diagnostics)
        line_clusters = {}
        for file_path, lines in line_distribution.items():
            if len(lines) > 1:
                lines.sort()
                clusters = []
                current_cluster = [lines[0]]

                for line in lines[1:]:
                    if line - current_cluster[-1] <= 5:  # Within 5 lines
                        current_cluster.append(line)
                    else:
                        if len(current_cluster) >= 2:
                            clusters.append(
                                {
                                    "start_line": min(current_cluster),
                                    "end_line": max(current_cluster),
                                    "diagnostic_count": len(current_cluster),
                                }
                            )
                        current_cluster = [line]

                if len(current_cluster) >= 2:
                    clusters.append(
                        {
                            "start_line": min(current_cluster),
                            "end_line": max(current_cluster),
                            "diagnostic_count": len(current_cluster),
                        }
                    )

                if clusters:
                    line_clusters[os.path.basename(file_path)] = clusters

        patterns_data = {
            "common_message_patterns": dict(message_patterns.most_common(10)),
            "common_diagnostic_codes": dict(code_patterns.most_common(10)),
            "line_clusters": line_clusters,
            "total_patterns_found": len(message_patterns) + len(code_patterns),
        }

        return APIResponse.success(patterns_data)
