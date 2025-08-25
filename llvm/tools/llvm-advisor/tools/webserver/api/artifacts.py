# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import os
import sys
from typing import Dict, Any
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
tools_dir = current_dir.parent.parent
sys.path.insert(0, str(tools_dir))

from .base import BaseEndpoint, APIResponse
from common.models import FileType


class ArtifactsEndpoint(BaseEndpoint):
    """GET /api/artifacts/{file_type} - Get aggregated data for a file type across all units"""

    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        if len(path_parts) < 3:
            return APIResponse.invalid_request("File type required")

        file_type_str = path_parts[2]

        # Validate file type
        try:
            file_type = FileType(file_type_str)
        except ValueError:
            return APIResponse.invalid_request(f"Invalid file type: {file_type_str}")

        parsed_data = self.get_parsed_data()

        # Aggregate data from all units for this file type
        aggregated_data = {
            "file_type": file_type.value,
            "units": {},
            "global_summary": {
                "total_files": 0,
                "total_errors": 0,
                "units_with_type": 0,
            },
        }

        for unit_name, unit_data in parsed_data.items():
            if file_type in unit_data:
                unit_files = []
                error_count = 0
                unit_summary_stats = {}

                for parsed_file in unit_data[file_type]:
                    has_error = "error" in parsed_file.metadata
                    if has_error:
                        error_count += 1

                    file_summary = {
                        "file_name": os.path.basename(parsed_file.file_path),
                        "file_path": parsed_file.file_path,
                        "file_size_bytes": parsed_file.metadata.get("file_size", 0),
                        "has_error": has_error,
                        "metadata": parsed_file.metadata,
                    }

                    # Include relevant summary data based on file type
                    if (
                        isinstance(parsed_file.data, dict)
                        and "summary" in parsed_file.data
                    ):
                        file_summary["summary"] = parsed_file.data["summary"]

                        # Aggregate numeric summary stats
                        for key, value in parsed_file.data["summary"].items():
                            if isinstance(value, (int, float)):
                                unit_summary_stats[key] = (
                                    unit_summary_stats.get(key, 0) + value
                                )

                    elif isinstance(parsed_file.data, list):
                        file_summary["item_count"] = len(parsed_file.data)
                        unit_summary_stats["total_items"] = unit_summary_stats.get(
                            "total_items", 0
                        ) + len(parsed_file.data)

                    unit_files.append(file_summary)

                aggregated_data["units"][unit_name] = {
                    "files": unit_files,
                    "count": len(unit_files),
                    "errors": error_count,
                    "summary_stats": unit_summary_stats,
                }

                aggregated_data["global_summary"]["total_files"] += len(unit_files)
                aggregated_data["global_summary"]["total_errors"] += error_count
                aggregated_data["global_summary"]["units_with_type"] += 1

        # Add file type specific aggregations
        if file_type_str == "diagnostics":
            aggregated_data["global_summary"]["total_diagnostics"] = sum(
                unit["summary_stats"].get("total_diagnostics", 0)
                for unit in aggregated_data["units"].values()
            )
        elif file_type_str == "remarks":
            aggregated_data["global_summary"]["total_remarks"] = sum(
                unit["summary_stats"].get("total_remarks", 0)
                for unit in aggregated_data["units"].values()
            )
        elif file_type_str in ["time-trace", "runtime-trace"]:
            aggregated_data["global_summary"]["total_events"] = sum(
                unit["summary_stats"].get("total_events", 0)
                for unit in aggregated_data["units"].values()
            )

        return APIResponse.success(aggregated_data)


class ArtifactTypesEndpoint(BaseEndpoint):
    """GET /api/artifacts - List all available artifact types with counts"""

    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        parsed_data = self.get_parsed_data()

        # Count files by type across all units
        type_counts = {}

        for unit_name, unit_data in parsed_data.items():
            for file_type, parsed_files in unit_data.items():
                type_name = file_type.value
                if type_name not in type_counts:
                    type_counts[type_name] = {
                        "total_files": 0,
                        "total_errors": 0,
                        "units": [],
                    }

                error_count = sum(1 for f in parsed_files if "error" in f.metadata)
                type_counts[type_name]["total_files"] += len(parsed_files)
                type_counts[type_name]["total_errors"] += error_count
                type_counts[type_name]["units"].append(unit_name)

        response_data = {
            "supported_types": [ft.value for ft in FileType],
            "available_types": type_counts,
            "total_types_found": len(type_counts),
        }

        return APIResponse.success(response_data)
