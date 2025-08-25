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
from typing import Dict, Any, List
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
tools_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(tools_dir))

from common.models import FileType, Remark
from ..base import APIResponse
from .base_specialized import BaseSpecializedEndpoint


class RemarksEndpoint(BaseSpecializedEndpoint):
    """Specialized endpoints for optimization remarks analysis"""

    def handle_overview(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/remarks/overview - Overall remarks statistics"""
        parsed_data = self.get_parsed_data()

        total_remarks = 0
        pass_distribution = Counter()
        function_distribution = Counter()
        location_distribution = defaultdict(int)

        for unit_name, unit_data in parsed_data.items():
            if FileType.REMARKS in unit_data:
                for parsed_file in unit_data[FileType.REMARKS]:
                    if isinstance(parsed_file.data, list):
                        total_remarks += len(parsed_file.data)

                        for remark in parsed_file.data:
                            if isinstance(remark, Remark):
                                pass_distribution[remark.pass_name] += 1
                                function_distribution[remark.function] += 1

                                if remark.location and remark.location.file:
                                    location_distribution[remark.location.file] += 1

        overview_data = {
            "totals": {
                "remarks": total_remarks,
                "unique_passes": len(pass_distribution),
                "unique_functions": len(function_distribution),
                "source_files": len(location_distribution),
            },
            "top_passes": dict(pass_distribution.most_common(10)),
            "top_functions": dict(function_distribution.most_common(10)),
            "top_files": dict(
                sorted(location_distribution.items(), key=lambda x: x[1], reverse=True)[
                    :10
                ]
            ),
        }

        return APIResponse.success(overview_data)

    def handle_passes(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/remarks/passes - Analysis by optimization passes"""
        parsed_data = self.get_parsed_data()

        passes_data = defaultdict(
            lambda: {"count": 0, "functions": set(), "files": set(), "examples": []}
        )

        for unit_name, unit_data in parsed_data.items():
            if FileType.REMARKS in unit_data:
                for parsed_file in unit_data[FileType.REMARKS]:
                    if isinstance(parsed_file.data, list):
                        for remark in parsed_file.data:
                            if isinstance(remark, Remark):
                                pass_name = remark.pass_name
                                passes_data[pass_name]["count"] += 1
                                passes_data[pass_name]["functions"].add(remark.function)

                                if remark.location and remark.location.file:
                                    passes_data[pass_name]["files"].add(
                                        remark.location.file
                                    )

                                # Keep a few examples
                                if len(passes_data[pass_name]["examples"]) < 3:
                                    example = {
                                        "function": remark.function,
                                        "message": (
                                            remark.message[:100] + "..."
                                            if len(remark.message) > 100
                                            else remark.message
                                        ),
                                        "location": {
                                            "file": (
                                                remark.location.file
                                                if remark.location
                                                else None
                                            ),
                                            "line": (
                                                remark.location.line
                                                if remark.location
                                                else None
                                            ),
                                        },
                                    }
                                    passes_data[pass_name]["examples"].append(example)

        # Convert sets to counts for JSON serialization
        result = {}
        for pass_name, data in passes_data.items():
            result[pass_name] = {
                "count": data["count"],
                "unique_functions": len(data["functions"]),
                "unique_files": len(data["files"]),
                "examples": data["examples"],
            }

        return APIResponse.success({"passes": result, "total_passes": len(result)})

    def handle_functions(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/remarks/functions - Analysis by functions"""
        parsed_data = self.get_parsed_data()

        functions_data = defaultdict(
            lambda: {
                "remarks_count": 0,
                "passes": set(),
                "locations": set(),
                "messages": [],
            }
        )

        for unit_name, unit_data in parsed_data.items():
            if FileType.REMARKS in unit_data:
                for parsed_file in unit_data[FileType.REMARKS]:
                    if isinstance(parsed_file.data, list):
                        for remark in parsed_file.data:
                            if isinstance(remark, Remark):
                                func_name = remark.function
                                functions_data[func_name]["remarks_count"] += 1
                                functions_data[func_name]["passes"].add(
                                    remark.pass_name
                                )

                                if remark.location:
                                    loc_str = (
                                        f"{remark.location.file}:{remark.location.line}"
                                    )
                                    functions_data[func_name]["locations"].add(loc_str)

                                # Keep sample messages
                                if len(functions_data[func_name]["messages"]) < 5:
                                    functions_data[func_name]["messages"].append(
                                        {
                                            "pass": remark.pass_name,
                                            "message": (
                                                remark.message[:150] + "..."
                                                if len(remark.message) > 150
                                                else remark.message
                                            ),
                                        }
                                    )

        # Convert to serializable format
        result = {}
        for func_name, data in functions_data.items():
            result[func_name] = {
                "remarks_count": data["remarks_count"],
                "unique_passes": len(data["passes"]),
                "unique_locations": len(data["locations"]),
                "passes": list(data["passes"]),
                "sample_messages": data["messages"],
            }

        # Sort by remarks count
        sorted_functions = dict(
            sorted(result.items(), key=lambda x: x[1]["remarks_count"], reverse=True)
        )

        return APIResponse.success(
            {"functions": sorted_functions, "total_functions": len(sorted_functions)}
        )

    def handle_hotspots(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/remarks/hotspots - Find optimization hotspots"""
        parsed_data = self.get_parsed_data()

        file_hotspots = defaultdict(
            lambda: {
                "remarks_count": 0,
                "line_distribution": defaultdict(int),
                "passes": set(),
                "functions": set(),
            }
        )

        for unit_name, unit_data in parsed_data.items():
            if FileType.REMARKS in unit_data:
                for parsed_file in unit_data[FileType.REMARKS]:
                    if isinstance(parsed_file.data, list):
                        for remark in parsed_file.data:
                            if (
                                isinstance(remark, Remark)
                                and remark.location
                                and remark.location.file
                            ):
                                file_path = remark.location.file
                                file_hotspots[file_path]["remarks_count"] += 1

                                if remark.location.line:
                                    file_hotspots[file_path]["line_distribution"][
                                        remark.location.line
                                    ] += 1

                                file_hotspots[file_path]["passes"].add(remark.pass_name)
                                file_hotspots[file_path]["functions"].add(
                                    remark.function
                                )

        # Convert to serializable format and find top hotspots
        hotspots = []
        for file_path, data in file_hotspots.items():
            hotspot = {
                "file": file_path,
                "file_name": os.path.basename(file_path),
                "remarks_count": data["remarks_count"],
                "unique_passes": len(data["passes"]),
                "unique_functions": len(data["functions"]),
                "hot_lines": dict(
                    sorted(
                        data["line_distribution"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                ),
            }
            hotspots.append(hotspot)

        # Sort by remarks count
        hotspots.sort(key=lambda x: x["remarks_count"], reverse=True)

        return APIResponse.success(
            {
                "hotspots": hotspots[:20],  # Top 20 hotspots
                "total_files_with_remarks": len(hotspots),
            }
        )
