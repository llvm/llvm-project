# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import sys
from collections import defaultdict, Counter
from typing import Dict, Any, List
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
tools_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(tools_dir))

from common.models import FileType, TraceEvent
from ..base import BaseEndpoint, APIResponse


class RuntimeTraceEndpoint(BaseEndpoint):
    """Specialized endpoints for runtime trace analysis (profile.json format)"""

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
        """GET /api/runtime-trace/overview - Overall runtime timing statistics"""
        parsed_data = self.get_parsed_data()

        total_events = 0
        event_categories = Counter()
        event_phases = Counter()
        duration_events = []

        for unit_name, unit_data in parsed_data.items():
            if FileType.RUNTIME_TRACE in unit_data:
                for parsed_file in unit_data[FileType.RUNTIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        total_events += len(parsed_file.data)

                        for event in parsed_file.data:
                            if isinstance(event, TraceEvent):
                                event_categories[event.category] += 1
                                event_phases[event.phase] += 1

                                if event.duration is not None:
                                    duration_events.append(event.duration)

        # Calculate timing statistics
        timing_stats = {}
        if duration_events:
            duration_events.sort()
            timing_stats = {
                "total_duration": sum(duration_events),
                "average_duration": sum(duration_events) / len(duration_events),
                "median_duration": duration_events[len(duration_events) // 2],
                "p95_duration": duration_events[int(len(duration_events) * 0.95)],
                "max_duration": max(duration_events),
                "events_with_duration": len(duration_events),
            }

        overview_data = {
            "totals": {
                "events": total_events,
                "categories": len(event_categories),
                "unique_phases": len(event_phases),
            },
            "timing_statistics": timing_stats,
            "category_distribution": dict(event_categories.most_common(10)),
            "phase_distribution": dict(event_phases.most_common(10)),
        }

        return APIResponse.success(overview_data)

    def handle_flamegraph(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/runtime-trace/flamegraph - Get flamegraph data for runtime time order view"""
        parsed_data = self.get_parsed_data()
        unit_name = query_params.get("unit", [None])[0]

        if unit_name and unit_name in parsed_data:
            unit_data = {unit_name: parsed_data[unit_name]}
        else:
            unit_data = parsed_data

        all_stacks = []

        for unit, data in unit_data.items():
            if FileType.RUNTIME_TRACE in data:
                for parsed_file in data[FileType.RUNTIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        from common.parsers.runtime_trace_parser import (
                            RuntimeTraceParser,
                        )

                        parser = RuntimeTraceParser()
                        flamegraph_data = parser.get_flamegraph_data(parsed_file.data)

                        for sample in flamegraph_data.get("samples", []):
                            sample["unit"] = unit
                            sample["source"] = "runtime"  # Mark as runtime data
                            all_stacks.append(sample)

        # Sort by timestamp for time order view
        all_stacks.sort(key=lambda x: x.get("timestamp", 0))

        return APIResponse.success(
            {"samples": all_stacks, "total_samples": len(all_stacks)}
        )

    def handle_sandwich(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/runtime-trace/sandwich - Get sandwich view data (aggregated by function)"""
        parsed_data = self.get_parsed_data()
        unit_name = query_params.get("unit", [None])[0]

        if unit_name and unit_name in parsed_data:
            unit_data = {unit_name: parsed_data[unit_name]}
        else:
            unit_data = parsed_data

        all_functions = []

        for unit, data in unit_data.items():
            if FileType.RUNTIME_TRACE in data:
                for parsed_file in data[FileType.RUNTIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        from common.parsers.runtime_trace_parser import (
                            RuntimeTraceParser,
                        )

                        parser = RuntimeTraceParser()
                        sandwich_data = parser.get_sandwich_data(parsed_file.data)

                        for func in sandwich_data.get("functions", []):
                            func["unit"] = unit
                            func["source"] = "runtime"  # Mark as runtime data
                            all_functions.append(func)

        # Merge functions with same name across units
        function_map = {}
        for func in all_functions:
            name = func["name"]
            if name not in function_map:
                function_map[name] = {
                    "name": name,
                    "total_time": 0,
                    "call_count": 0,
                    "category": func.get("category", "Runtime"),
                    "units": [],
                    "source": "runtime",
                }

            function_map[name]["total_time"] += func["total_time"]
            function_map[name]["call_count"] += func["call_count"]
            function_map[name]["units"].append(func["unit"])

        # Calculate averages and sort
        merged_functions = []
        for func_data in function_map.values():
            if func_data["call_count"] > 0:
                func_data["avg_time"] = (
                    func_data["total_time"] / func_data["call_count"]
                )
            else:
                func_data["avg_time"] = 0
            merged_functions.append(func_data)

        merged_functions.sort(key=lambda x: x["total_time"], reverse=True)

        return APIResponse.success(
            {"functions": merged_functions, "total_functions": len(merged_functions)}
        )

    def handle_hotspots(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/runtime-trace/hotspots - Find runtime performance hotspots"""
        parsed_data = self.get_parsed_data()

        event_durations = defaultdict(list)
        category_times = defaultdict(float)
        thread_times = defaultdict(float)

        for unit_name, unit_data in parsed_data.items():
            if FileType.RUNTIME_TRACE in unit_data:
                for parsed_file in unit_data[FileType.RUNTIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        for event in parsed_file.data:
                            if (
                                isinstance(event, TraceEvent)
                                and event.duration is not None
                            ):
                                event_durations[event.name].append(event.duration)
                                category_times[
                                    event.category or "Runtime"
                                ] += event.duration

                                if event.tid is not None:
                                    thread_times[event.tid] += event.duration

        # Find hotspots by event name
        event_hotspots = []
        for event_name, durations in event_durations.items():
            if durations:
                total_time = sum(durations)
                event_hotspots.append(
                    {
                        "event_name": event_name,
                        "total_time": total_time,
                        "average_time": total_time / len(durations),
                        "max_time": max(durations),
                        "occurrences": len(durations),
                        "percentage_of_total": 0,  # Will be calculated below
                        "source": "runtime",
                    }
                )

        # Sort by total time
        event_hotspots.sort(key=lambda x: x["total_time"], reverse=True)

        # Calculate percentages
        total_trace_time = sum(h["total_time"] for h in event_hotspots)
        for hotspot in event_hotspots:
            hotspot["percentage_of_total"] = (
                hotspot["total_time"] / max(total_trace_time, 1)
            ) * 100

        hotspots_data = {
            "event_hotspots": event_hotspots[:20],  # Top 20
            "category_hotspots": dict(
                sorted(category_times.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "thread_hotspots": dict(
                sorted(thread_times.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "total_trace_time": total_trace_time,
        }

        return APIResponse.success(hotspots_data)
