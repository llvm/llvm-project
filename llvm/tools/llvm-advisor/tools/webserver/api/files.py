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


class FileContentEndpoint(BaseEndpoint):
    """GET /api/file/{unit_name}/{file_type}/{file_name} - Get parsed content of a specific file"""

    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        if len(path_parts) < 5:
            return APIResponse.invalid_request(
                "File path must include unit_name, file_type, and file_name"
            )

        unit_name = path_parts[2]
        file_type_str = path_parts[3]
        file_name = path_parts[4]

        # Validate file type
        try:
            file_type = FileType(file_type_str)
        except ValueError:
            return APIResponse.invalid_request(f"Invalid file type: {file_type_str}")

        parsed_data = self.get_parsed_data()

        # Validate unit exists
        if unit_name not in parsed_data:
            return APIResponse.not_found(f"Compilation unit '{unit_name}'")

        # Validate file type exists in unit
        if file_type not in parsed_data[unit_name]:
            return APIResponse.not_found(
                f"File type '{file_type_str}' in unit '{unit_name}'"
            )

        # Find the specific file
        target_file = None
        for parsed_file in parsed_data[unit_name][file_type]:
            if os.path.basename(parsed_file.file_path) == file_name:
                target_file = parsed_file
                break

        if not target_file:
            return APIResponse.not_found(
                f"File '{file_name}' of type '{file_type_str}' in unit '{unit_name}'"
            )

        # Check for streaming/partial parsing
        include_full_data = query_params.get("full", ["false"])[0].lower() == "true"

        # For code artifacts (assembly, ir, preprocessed, etc.), return raw file content
        code_artifact_types = {
            FileType.ASSEMBLY,
            FileType.IR,
            FileType.PREPROCESSED,
            FileType.MACRO_EXPANSION,
            FileType.AST_JSON,
        }

        if file_type in code_artifact_types:
            # Return raw file content for code viewing
            try:
                with open(target_file.file_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()

                response_data = {
                    "file_path": target_file.file_path,
                    "file_name": file_name,
                    "unit_name": unit_name,
                    "file_type": file_type.value,
                    "content": raw_content,
                    "data_type": "raw",
                    "metadata": target_file.metadata,
                    "has_error": "error" in target_file.metadata,
                }

                return APIResponse.success(response_data)

            except Exception as e:
                return APIResponse.server_error(
                    f"Failed to read file content: {str(e)}"
                )

        response_data = {
            "file_path": target_file.file_path,
            "file_name": file_name,
            "unit_name": unit_name,
            "file_type": file_type.value,
            "metadata": target_file.metadata,
            "has_error": "error" in target_file.metadata,
        }

        # Include data based on query parameters and file size
        if target_file.metadata.get("is_partial", False) and not include_full_data:
            # For large files that were partially parsed, provide summary only
            if isinstance(target_file.data, dict) and "summary" in target_file.data:
                response_data["summary"] = target_file.data["summary"]
                response_data["data_type"] = "summary"
            else:
                response_data["data"] = target_file.data
                response_data["data_type"] = "partial"
        else:
            response_data["data"] = target_file.data
            response_data["data_type"] = "full"

        return APIResponse.success(response_data)
