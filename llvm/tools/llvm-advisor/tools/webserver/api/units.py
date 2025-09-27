# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import os
import json
from typing import Dict, Any, List, Optional
from .base import BaseEndpoint, APIResponse


class UnitsEndpoint(BaseEndpoint):
    """GET /api/units - List all compilation units with basic info"""

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load compilation unit metadata from the C++ tracking system"""
        try:
            metadata_path = os.path.join(self.data_dir, ".llvm-advisor-metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")
        return None

    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        units = self.get_compilation_units()
        metadata = self._load_metadata()
        metadata_units = {}

        if metadata and "units" in metadata:
            for unit_meta in metadata["units"]:
                metadata_units[unit_meta["name"]] = unit_meta

        unit_list = []

        for unit in units:
            unit_info = {
                "name": unit.name,
                "path": unit.path,
                "artifact_types": [ft.value for ft in unit.artifacts.keys()],
                "artifact_counts": {
                    ft.value: len(files) for ft, files in unit.artifacts.items()
                },
                "total_files": sum(len(files) for files in unit.artifacts.values()),
            }

            if hasattr(unit, "metadata") and unit.metadata:
                if "run_timestamp" in unit.metadata:
                    unit_info["run_timestamp"] = unit.metadata["run_timestamp"]
                if "available_runs" in unit.metadata:
                    unit_info["available_runs"] = unit.metadata["available_runs"]
                if "run_path" in unit.metadata:
                    unit_info["run_path"] = unit.metadata["run_path"]

            if unit.name in metadata_units:
                unit_meta = metadata_units[unit.name]
                unit_info.update(
                    {
                        "metadata_timestamp": unit_meta.get("timestamp"),
                        "status": unit_meta.get("status", "unknown"),
                        "metadata": {
                            "output_path": unit_meta.get("output_path"),
                            "properties": unit_meta.get("properties", {}),
                        },
                    }
                )
            else:
                unit_info.update(
                    {"timestamp": None, "status": "unknown", "metadata": {}}
                )

            unit_list.append(unit_info)

        if metadata_units:
            unit_list.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        response_data = {
            "units": unit_list,
            "total_units": len(unit_list),
            "total_files": sum(unit["total_files"] for unit in unit_list),
        }

        return APIResponse.success(response_data)


class UnitDetailEndpoint(BaseEndpoint):
    """GET /api/units/{unit_name} - Detailed information for a specific compilation unit"""

    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        if len(path_parts) < 3:
            return APIResponse.invalid_request("Unit name required")

        unit_name = path_parts[2]
        parsed_data = self.get_parsed_data()

        if unit_name not in parsed_data:
            return APIResponse.not_found(f"Compilation unit '{unit_name}'")

        unit_data = parsed_data[unit_name]

        response = {
            "unit_name": unit_name,
            "artifact_types": {},
            "summary": {
                "total_artifact_types": len(unit_data),
                "total_files": 0,
                "total_errors": 0,
            },
        }

        for file_type, parsed_files in unit_data.items():
            file_list = []
            error_count = 0

            for parsed_file in parsed_files:
                has_error = "error" in parsed_file.metadata
                if has_error:
                    error_count += 1

                file_info = {
                    "file_path": parsed_file.file_path,
                    "file_name": os.path.basename(parsed_file.file_path),
                    "file_size_bytes": parsed_file.metadata.get("file_size", 0),
                    "has_error": has_error,
                    "metadata": parsed_file.metadata,
                }

                if isinstance(parsed_file.data, dict) and "summary" in parsed_file.data:
                    file_info["summary"] = parsed_file.data["summary"]
                elif isinstance(parsed_file.data, list):
                    file_info["item_count"] = len(parsed_file.data)

                file_list.append(file_info)

            response["artifact_types"][file_type.value] = {
                "files": file_list,
                "count": len(file_list),
                "errors": error_count,
            }

            response["summary"]["total_files"] += len(file_list)
            response["summary"]["total_errors"] += error_count

        return APIResponse.success(response)
