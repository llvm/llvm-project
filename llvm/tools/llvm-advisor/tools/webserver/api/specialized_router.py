# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
tools_dir = current_dir.parent.parent
sys.path.insert(0, str(tools_dir))

from common.collector import ArtifactCollector
from .base import APIResponse

# Import specialized endpoints
from .specialized.remarks_api import RemarksEndpoint
from .specialized.diagnostics_api import DiagnosticsEndpoint
from .specialized.compilation_phases_api import CompilationPhasesEndpoint
from .specialized.time_trace_api import TimeTraceEndpoint
from .specialized.runtime_trace_api import RuntimeTraceEndpoint
from .specialized.binary_size_api import BinarySizeEndpoint


class SpecializedRouter:
    """Router for specialized file-type specific endpoints"""

    def __init__(self, data_dir: str, collector: ArtifactCollector):
        self.data_dir = data_dir
        self.collector = collector

        # Initialize specialized endpoints
        self.endpoints = {
            "remarks": RemarksEndpoint(data_dir, collector),
            "diagnostics": DiagnosticsEndpoint(data_dir, collector),
            "compilation-phases": CompilationPhasesEndpoint(data_dir, collector),
            "time-trace": TimeTraceEndpoint(data_dir, collector),
            "runtime-trace": RuntimeTraceEndpoint(data_dir, collector),
            "binary-size": BinarySizeEndpoint(data_dir, collector),
        }

    def route_request(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """Route specialized requests to appropriate handlers"""

        if len(path_parts) < 2:
            return APIResponse.invalid_request("File type required")

        file_type = path_parts[1]

        if file_type not in self.endpoints:
            return APIResponse.error(
                f"Specialized endpoint not available for '{file_type}'", 404
            )

        endpoint = self.endpoints[file_type]

        # Determine sub-endpoint
        if len(path_parts) >= 3:
            sub_endpoint = path_parts[2]
        else:
            sub_endpoint = "overview"  # Default to overview

        # Route to specific handler method
        method_name = f"handle_{sub_endpoint.replace('-', '_')}"

        if hasattr(endpoint, method_name):
            handler_method = getattr(endpoint, method_name)
            return handler_method(path_parts, query_params)
        else:
            available_methods = [
                method[7:].replace("_", "-")
                for method in dir(endpoint)
                if method.startswith("handle_") and not method.startswith("handle__")
            ]

            return APIResponse.error(
                f"Sub-endpoint '{sub_endpoint}' not available for '{file_type}'. "
                f"Available: {', '.join(available_methods)}",
                404,
            )

    def get_available_endpoints(self) -> Dict[str, Dict[str, str]]:
        """Get all available specialized endpoints"""
        endpoints_info = {}

        for file_type, endpoint in self.endpoints.items():
            # Get all handler methods
            handlers = [
                method[7:].replace("_", "-")
                for method in dir(endpoint)
                if method.startswith("handle_") and not method.startswith("handle__")
            ]

            endpoints_info[file_type] = {
                "base_path": f"/api/{file_type}",
                "available_endpoints": handlers,
                "examples": [f"/api/{file_type}/{handler}" for handler in handlers[:3]],
            }

        return endpoints_info


# Endpoint documentation for each file type
SPECIALIZED_ENDPOINTS_DOCS = {
    "remarks": {
        "overview": "Overall optimization remarks statistics and distribution",
        "passes": "Analysis grouped by optimization passes",
        "functions": "Analysis grouped by functions with remarks",
        "hotspots": "Find files and locations with most optimization activity",
    },
    "diagnostics": {
        "overview": "Overall compiler diagnostics statistics",
        "by-level": "Analysis by diagnostic levels (error, warning, note)",
        "files": "Analysis by files with issues",
        "patterns": "Common diagnostic patterns and trends",
    },
    "compilation-phases": {
        "overview": "Overall compilation timing statistics",
        "phases": "Detailed analysis by individual compilation phases",
        "bottlenecks": "Identify compilation bottlenecks and slow phases",
        "trends": "Compilation time trends and consistency analysis",
    },
    "time-trace": {
        "overview": "Overall timing statistics from Chrome trace format",
        "timeline": "Timeline analysis of compilation events",
        "hotspots": "Find performance hotspots and slow operations",
        "categories": "Analysis by event categories",
        "parallelism": "Analyze parallelism and thread utilization",
    },
    "runtime-trace": {
        "overview": "Runtime profiling statistics",
        "timeline": "Runtime event timeline analysis",
        "hotspots": "Runtime performance hotspots",
        "categories": "Runtime event categories",
        "parallelism": "Runtime parallelism analysis",
    },
    "binary-size": {
        "overview": "Overall binary size statistics and breakdown",
        "sections": "Detailed analysis by binary sections",
        "optimization": "Size optimization opportunities and recommendations",
        "comparison": "Compare sizes across compilation units",
    },
}
