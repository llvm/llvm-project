# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import os
import sys
from typing import Dict, Any, List
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
tools_dir = current_dir.parent.parent
sys.path.insert(0, str(tools_dir))

from .base import BaseEndpoint, APIResponse
from common.models import FileType


class ExplorerEndpoint(BaseEndpoint):
    """Explorer endpoints for source code and artifact viewing"""

    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        """Route explorer requests to specific handlers"""
        if len(path_parts) < 3:
            return APIResponse.invalid_request("Explorer endpoint required")

        endpoint = path_parts[2]

        if endpoint == "files":
            return self.handle_files(path_parts, query_params)
        elif endpoint == "source":
            return self.handle_source(path_parts, query_params)
        elif endpoint in [
            "assembly",
            "ir",
            "optimized-ir",
            "object",
            "ast-json",
            "preprocessed",
            "macro-expansion",
        ]:
            return self.handle_artifact(path_parts, query_params, endpoint)
        else:
            return APIResponse.not_found(f"Explorer endpoint '{endpoint}'")

    def handle_files(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/explorer/files - Get available source files and their artifacts"""
        try:
            parsed_data = self.get_parsed_data()

            # Check if unit parameter is provided
            unit_filter = query_params.get("unit", [None])[0]

            if not unit_filter:
                return APIResponse.invalid_request("Unit parameter is required")

            # Only process the specified unit
            if unit_filter not in parsed_data:
                return APIResponse.not_found(f"Unit '{unit_filter}' not found")

            unit_artifacts = parsed_data[unit_filter]
            files_info = []

            print(f"Processing files for unit: {unit_filter}")

            # Get available artifact types for this unit
            available_types = set(unit_artifacts.keys())

            # PRIORITY: Get source files from collected sources (timestamped)
            source_files = set()

            # Use directly collected source files
            if FileType.SOURCES in unit_artifacts:
                for parsed_file in unit_artifacts[FileType.SOURCES]:
                    # Add collected source files
                    source_file_path = os.path.basename(parsed_file.file_path)
                    source_files.add(source_file_path)

            # Fallback to dependencies parsing
            if not source_files:
                if FileType.DEPENDENCIES in unit_artifacts:
                    for parsed_file in unit_artifacts[FileType.DEPENDENCIES]:
                        if isinstance(parsed_file.data, list):
                            for dependency in parsed_file.data:
                                # Handle both object-style and dict-style dependencies
                                source_path = None
                                target_path = None

                                if hasattr(dependency, "source"):
                                    source_path = dependency.source
                                elif (
                                    isinstance(dependency, dict)
                                    and "source" in dependency
                                ):
                                    source_path = dependency["source"]

                                if hasattr(dependency, "target"):
                                    target_path = dependency.target
                                elif (
                                    isinstance(dependency, dict)
                                    and "target" in dependency
                                ):
                                    target_path = dependency["target"]

                                # Filter for actual source files (skip object files, etc.)
                                if source_path and self._is_source_file(source_path):
                                    source_files.add(source_path)

                                if target_path and self._is_source_file(target_path):
                                    source_files.add(target_path)

            # Extract source file references from diagnostics and remarks as fallback
            for file_type in [FileType.DIAGNOSTICS, FileType.REMARKS]:
                if file_type in unit_artifacts:
                    for parsed_file in unit_artifacts[file_type]:
                        if isinstance(parsed_file.data, list):
                            for item in parsed_file.data:
                                if isinstance(item, dict) and "file" in item:
                                    source_path = item["file"]
                                    if self._is_source_file(source_path):
                                        source_files.add(source_path)

            # For each identified source file, check what artifacts are available
            for source_file in source_files:
                available_artifacts = self._get_available_artifacts_for_source(
                    source_file, unit_artifacts
                )

                if available_artifacts:  # Only include files that have artifacts
                    # Make the path relative to make it cleaner for display
                    display_path = source_file
                    if source_file.startswith("./"):
                        display_path = source_file[2:]

                    files_info.append(
                        {
                            "path": source_file,
                            "name": os.path.basename(source_file),
                            "display_name": display_path,
                            "unit": unit_filter,  # Use unit_filter instead of unit_name
                            "available_artifacts": available_artifacts,
                        }
                    )

            # Remove duplicates and sort
            unique_files = {}
            for file_info in files_info:
                key = file_info["path"]
                if key not in unique_files:
                    unique_files[key] = file_info
                else:
                    # Merge available artifacts
                    existing_artifacts = set(unique_files[key]["available_artifacts"])
                    new_artifacts = set(file_info["available_artifacts"])
                    unique_files[key]["available_artifacts"] = list(
                        existing_artifacts | new_artifacts
                    )

            final_files = sorted(unique_files.values(), key=lambda x: x["display_name"])

            return APIResponse.success(
                {"files": final_files, "count": len(final_files)}
            )

        except Exception as e:
            return APIResponse.error(f"Failed to load source files: {str(e)}")

    def handle_source(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/explorer/source/{file_path} - Get source code content"""
        if len(path_parts) < 4:
            return APIResponse.invalid_request("File path required")

        file_path = "/".join(path_parts[3:])

        try:
            # Look for source files in the timestamped runs first
            source_content = None
            parsed_data = self.get_parsed_data()

            # Check parsed sources from timestamped runs
            for unit_name, unit_artifacts in parsed_data.items():
                if FileType.SOURCES in unit_artifacts:
                    for parsed_file in unit_artifacts[FileType.SOURCES]:
                        # Extract filename from both the requested path and stored path
                        requested_filename = os.path.basename(file_path)
                        stored_filename = os.path.basename(parsed_file.file_path)

                        if requested_filename == stored_filename:
                            # Found the source file in our collected sources
                            if os.path.exists(parsed_file.file_path):
                                with open(
                                    parsed_file.file_path,
                                    "r",
                                    encoding="utf-8",
                                    errors="ignore",
                                ) as f:
                                    source_content = f.read()
                                    break

                if source_content:
                    break

            # Fallback to original filesystem search
            if not source_content:
                # Try absolute path first
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        source_content = f.read()

                # Try relative to data directory (where .llvm-advisor is)
                if not source_content:
                    # Remove leading ./ if present
                    clean_path = file_path
                    if clean_path.startswith("./"):
                        clean_path = clean_path[2:]

                    relative_path = os.path.join(self.data_dir, clean_path)
                    if os.path.exists(relative_path):
                        with open(
                            relative_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            source_content = f.read()

                # Try one level up from data directory (since source files might be outside .llvm-advisor)
                if not source_content:
                    parent_dir = os.path.dirname(self.data_dir)
                    clean_path = file_path
                    if clean_path.startswith("./"):
                        clean_path = clean_path[2:]

                    parent_relative_path = os.path.join(parent_dir, clean_path)
                    if os.path.exists(parent_relative_path):
                        with open(
                            parent_relative_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            source_content = f.read()

            if not source_content:
                return APIResponse.not_found(f"Source file '{file_path}'")

            # Get inline data (diagnostics, remarks) from parsed data
            inline_data = self._get_inline_data_for_file(parsed_data, file_path)

            return APIResponse.success(
                {
                    "source": source_content,
                    "file_path": file_path,
                    "language": self._detect_language(file_path),
                    "inline_data": inline_data,
                }
            )

        except Exception as e:
            return APIResponse.error(f"Failed to load source file: {str(e)}")

    def handle_artifact(
        self, path_parts: list, query_params: Dict[str, list], artifact_type: str
    ) -> Dict[str, Any]:
        """GET /api/explorer/{artifact_type}/{file_path} - Get artifact content"""
        if len(path_parts) < 4:
            return APIResponse.invalid_request("File path required")

        file_path = "/".join(path_parts[3:])

        try:
            parsed_data = self.get_parsed_data()

            # Map endpoint names to FileType enums
            type_mapping = {
                "assembly": FileType.ASSEMBLY,
                "ir": FileType.IR,
                "optimized-ir": FileType.IR,
                "object": FileType.OBJDUMP,
                "ast-json": FileType.AST_JSON,
                "preprocessed": FileType.PREPROCESSED,
                "macro-expansion": FileType.MACRO_EXPANSION,
            }

            if artifact_type not in type_mapping:
                return APIResponse.invalid_request(
                    f"Unknown artifact type: {artifact_type}"
                )

            file_type = type_mapping[artifact_type]
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Try to read the raw file directly from .llvm-advisor
            raw_content = self._try_read_raw_artifact(
                file_path, artifact_type, base_name
            )
            if raw_content:
                return APIResponse.success(
                    {
                        "content": raw_content,
                        "file_path": file_path,
                        "artifact_type": artifact_type,
                    }
                )

            # Fall back to parsed data if raw file not found
            content = None
            for unit_name, unit_artifacts in parsed_data.items():
                if file_type in unit_artifacts:
                    for parsed_file in unit_artifacts[file_type]:
                        artifact_base = os.path.splitext(
                            os.path.basename(parsed_file.file_path)
                        )[0]
                        # More flexible matching to ensure we find the right content
                        if (
                            base_name in artifact_base
                            or artifact_base in base_name
                            or self._matches_artifact_to_source(
                                parsed_file.file_path, file_path
                            )
                        ):
                            # Try to read the raw file directly based on parsed_file.file_path
                            if os.path.exists(parsed_file.file_path):
                                try:
                                    with open(
                                        parsed_file.file_path,
                                        "r",
                                        encoding="utf-8",
                                        errors="ignore",
                                    ) as f:
                                        raw_file_content = f.read()
                                        content = raw_file_content
                                        break
                                except Exception as e:
                                    pass

                            # Use the already parsed content as fallback
                            if not content:
                                content = self._format_parsed_content(parsed_file.data)
                            if (
                                content
                                and content.strip()
                                and content.strip() != "# No data available"
                            ):
                                break
                    if (
                        content
                        and content.strip()
                        and content.strip() != "# No data available"
                    ):
                        break

            if not content:
                return APIResponse.not_found(f"{artifact_type} for file '{file_path}'")

            return APIResponse.success(
                {
                    "content": content,
                    "file_path": file_path,
                    "artifact_type": artifact_type,
                }
            )

        except Exception as e:
            return APIResponse.error(f"Failed to load {artifact_type}: {str(e)}")

    def _format_parsed_content(self, data: Any) -> str:
        """Format already parsed content for display"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if "content" in data:
                return data["content"]
            elif "text" in data:
                return data["text"]
            elif "data" in data:
                # Sometimes the actual content is nested in a 'data' field
                return self._format_parsed_content(data["data"])
            elif "instructions" in data:
                # Handle assembly with instructions field
                if isinstance(data["instructions"], list):
                    return "\n".join(str(inst) for inst in data["instructions"])
                else:
                    return str(data["instructions"])
            elif "assembly" in data:
                # Handle assembly content
                return str(data["assembly"])
            elif "ir" in data:
                # Handle LLVM IR content
                return str(data["ir"])
            elif "source" in data:
                # Handle source content
                return str(data["source"])
            else:
                # For structured data, format as JSON with proper indentation
                import json

                return json.dumps(data, indent=2)
        elif isinstance(data, list):
            # Handle lists - could be lines of code, assembly instructions, etc.
            if len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, str):
                    # Simple list of strings - join with newlines
                    return "\n".join(str(item) for item in data)
                elif isinstance(first_item, dict):
                    # List of structured data - try to extract meaningful content
                    lines = []
                    for item in data:
                        if isinstance(item, dict):
                            # Try to extract text/content/instruction from each item
                            if "instruction" in item:
                                lines.append(str(item["instruction"]))
                            elif "text" in item:
                                lines.append(str(item["text"]))
                            elif "content" in item:
                                lines.append(str(item["content"]))
                            elif "line" in item:
                                lines.append(str(item["line"]))
                            elif "assembly" in item:
                                lines.append(str(item["assembly"]))
                            elif "ir" in item:
                                lines.append(str(item["ir"]))
                            else:
                                # For complex objects, try to stringify them meaningfully
                                if hasattr(item, "__str__") and not isinstance(
                                    item, dict
                                ):
                                    lines.append(str(item))
                                else:
                                    # Extract all values and join them
                                    values = [
                                        str(v) for v in item.values() if v is not None
                                    ]
                                    if values:
                                        lines.append(" ".join(values))
                                    else:
                                        lines.append(json.dumps(item, indent=2))
                        else:
                            lines.append(str(item))
                    return "\n".join(lines)
                else:
                    return "\n".join(str(item) for item in data)
            else:
                return "# No data available"
        else:
            # Handle any other data types including custom objects
            if hasattr(data, "__dict__"):
                # For custom objects, try to extract meaningful content
                return str(data)
            else:
                return str(data)

    def _get_inline_data_for_file(
        self, parsed_data: Dict, file_path: str
    ) -> Dict[str, List[Dict]]:
        """Get inline data (diagnostics, remarks) for a specific file using parsed data"""
        inline_data = {"diagnostics": [], "remarks": []}

        for unit_name, unit_artifacts in parsed_data.items():
            # Get diagnostics
            if FileType.DIAGNOSTICS in unit_artifacts:
                for parsed_file in unit_artifacts[FileType.DIAGNOSTICS]:
                    if isinstance(parsed_file.data, list):
                        for diagnostic in parsed_file.data:
                            # Handle both dataclass objects and dictionaries
                            if hasattr(diagnostic, "location") and diagnostic.location:
                                if self._matches_file(
                                    diagnostic.location.file or "", file_path
                                ):
                                    inline_data["diagnostics"].append(
                                        {
                                            "line": diagnostic.location.line or 0,
                                            "column": diagnostic.location.column or 0,
                                            "level": diagnostic.level,
                                            "message": diagnostic.message,
                                            "type": "diagnostic",
                                        }
                                    )
                            elif isinstance(diagnostic, dict) and self._matches_file(
                                diagnostic.get("file", ""), file_path
                            ):
                                inline_data["diagnostics"].append(
                                    {
                                        "line": diagnostic.get("line", 0),
                                        "column": diagnostic.get("column", 0),
                                        "level": diagnostic.get("level", "info"),
                                        "message": diagnostic.get("message", ""),
                                        "type": "diagnostic",
                                    }
                                )

            # Get remarks
            if FileType.REMARKS in unit_artifacts:
                for parsed_file in unit_artifacts[FileType.REMARKS]:
                    if isinstance(parsed_file.data, list):
                        for remark in parsed_file.data:
                            # Handle both dataclass objects and dictionaries
                            if hasattr(remark, "location") and remark.location:
                                if self._matches_file(
                                    remark.location.file or "", file_path
                                ):
                                    inline_data["remarks"].append(
                                        {
                                            "line": remark.location.line or 0,
                                            "column": remark.location.column or 0,
                                            "level": "remark",
                                            "message": remark.message,
                                            "pass": remark.pass_name,
                                            "type": "remark",
                                        }
                                    )
                            elif isinstance(remark, dict) and self._matches_file(
                                remark.get("file", ""), file_path
                            ):
                                inline_data["remarks"].append(
                                    {
                                        "line": remark.get("line", 0),
                                        "column": remark.get("column", 0),
                                        "level": remark.get("level", "info"),
                                        "message": remark.get("message", ""),
                                        "pass": remark.get("pass", ""),
                                        "type": "remark",
                                    }
                                )

        return inline_data

    def _matches_file(self, path1: str, path2: str) -> bool:
        """Check if two paths refer to the same source file"""
        if not path1 or not path2:
            return False

        # Normalize paths
        norm1 = os.path.normpath(path1).replace("\\", "/")
        norm2 = os.path.normpath(path2).replace("\\", "/")

        # Direct match or basename match
        return (
            norm1 == norm2
            or os.path.basename(norm1) == os.path.basename(norm2)
            or norm1.endswith(norm2)
            or norm2.endswith(norm1)
        )

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = os.path.splitext(file_path)[1].lower()

        lang_map = {
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c++": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".hxx": "cpp",
            ".h++": "cpp",
            ".py": "python",
            ".js": "javascript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
        }

        return lang_map.get(ext, "text")

    def _is_source_file(self, file_path: str) -> bool:
        """Check if a file path represents a source code file"""
        if not file_path:
            return False

        # Get file extension
        ext = os.path.splitext(file_path)[1].lower()

        # List of source file extensions
        source_extensions = {
            ".c",
            ".cpp",
            ".cc",
            ".cxx",
            ".c++",
            ".h",
            ".hpp",
            ".hxx",
            ".h++",
            ".py",
            ".js",
            ".rs",
            ".go",
            ".java",
            ".swift",
            ".kt",
            ".scala",
            ".rb",
            ".php",
            ".pl",
            ".sh",
            ".bash",
        }

        return ext in source_extensions

    def _get_available_artifacts_for_source(
        self, source_file_path: str, unit_artifacts: Dict
    ) -> List[str]:
        """Check which artifacts are available for a given source file"""
        available_artifacts = []
        base_name = os.path.splitext(os.path.basename(source_file_path))[0]

        # Map artifact types to their display names
        type_map = {
            FileType.ASSEMBLY: "assembly",
            FileType.IR: "ir",
            FileType.AST_JSON: "ast-json",
            FileType.OBJDUMP: "object",
            FileType.PREPROCESSED: "preprocessed",
        }

        for file_type, display_name in type_map.items():
            if file_type in unit_artifacts:
                # Check if this source file has this type of artifact
                for parsed_file in unit_artifacts[file_type]:
                    artifact_base = os.path.splitext(
                        os.path.basename(parsed_file.file_path)
                    )[0]
                    # Try various matching strategies
                    if (
                        artifact_base == base_name
                        or base_name in artifact_base
                        or artifact_base in base_name
                        or self._matches_artifact_to_source(
                            parsed_file.file_path, source_file_path
                        )
                    ):
                        available_artifacts.append(display_name)
                        break

        return available_artifacts

    def _try_read_raw_artifact(
        self, file_path: str, artifact_type: str, base_name: str
    ) -> str:
        """Try to read the raw artifact file directly from .llvm-advisor directory"""
        try:
            # Map artifact types to their directory names and file extensions
            artifact_mapping = {
                "assembly": ("assembly", [".s", ".asm"]),
                "ir": ("ir", [".ll"]),
                "optimized-ir": ("ir", [".ll"]),
                "object": ("objdump", [".objdump", ".obj", ".txt"]),
                "ast-json": ("ast-json", [".json"]),
                "preprocessed": ("preprocessed", [".i", ".ii"]),
                "macro-expansion": ("macro-expansion", [".i", ".ii"]),
            }

            if artifact_type not in artifact_mapping:
                return None

            dir_name, extensions = artifact_mapping[artifact_type]

            # Look for the artifact file in all compilation units
            for root, dirs, files in os.walk(self.data_dir):
                if dir_name in os.path.basename(root):
                    # Try to find files that match our base name
                    for file in files:
                        file_base = os.path.splitext(file)[0]
                        file_ext = os.path.splitext(file)[1]
                        if (
                            file_base == base_name
                            or base_name in file_base
                            or file_base in base_name
                        ):
                            # Check if file has the right extension
                            if file_ext in extensions or not extensions:
                                artifact_path = os.path.join(root, file)
                                with open(
                                    artifact_path,
                                    "r",
                                    encoding="utf-8",
                                    errors="ignore",
                                ) as f:
                                    content = f.read()
                                    return content
            return None

        except Exception as e:
            return None

    def _matches_artifact_to_source(self, artifact_path: str, source_path: str) -> bool:
        """Check if an artifact matches a source file using various heuristics"""
        artifact_name = os.path.basename(artifact_path)
        source_name = os.path.basename(source_path)

        # Remove extensions for comparison
        artifact_base = os.path.splitext(artifact_name)[0]
        source_base = os.path.splitext(source_name)[0]

        # Direct match
        if artifact_base == source_base:
            return True

        # Handle cases like test.cpp -> test.s, test.ll, etc.
        if artifact_base.startswith(source_base) or source_base.startswith(
            artifact_base
        ):
            return True

        # Handle mangled names or similar patterns
        # This could be expanded based on actual file naming patterns found
        return False
