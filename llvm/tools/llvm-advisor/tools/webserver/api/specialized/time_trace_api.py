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


class TimeTraceEndpoint(BaseEndpoint):
    """Specialized endpoints for time trace analysis (Chrome trace format)"""

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
        """GET /api/time-trace/overview - Overall timing statistics"""
        parsed_data = self.get_parsed_data()

        total_events = 0
        event_categories = Counter()
        event_phases = Counter()
        duration_events = []

        for unit_name, unit_data in parsed_data.items():
            if FileType.TIME_TRACE in unit_data:
                for parsed_file in unit_data[FileType.TIME_TRACE]:
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

    def handle_timeline(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/time-trace/timeline - Timeline analysis of events"""
        parsed_data = self.get_parsed_data()

        timeline_data = []
        processes = set()
        threads = set()

        for unit_name, unit_data in parsed_data.items():
            if FileType.TIME_TRACE in unit_data:
                for parsed_file in unit_data[FileType.TIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        for event in parsed_file.data:
                            if isinstance(event, TraceEvent):
                                if event.pid is not None:
                                    processes.add(event.pid)
                                if event.tid is not None:
                                    threads.add(event.tid)

                                timeline_entry = {
                                    "unit": unit_name,
                                    "timestamp": event.timestamp,
                                    "name": event.name,
                                    "category": event.category,
                                    "phase": event.phase,
                                    "duration": event.duration,
                                    "pid": event.pid,
                                    "tid": event.tid,
                                    "args": event.args if event.args else {},
                                }
                                timeline_data.append(timeline_entry)

        # Sort by timestamp
        timeline_data.sort(key=lambda x: x["timestamp"])

        # Limit to reasonable size for API response
        max_events = int(query_params.get("limit", ["1000"])[0])
        timeline_data = timeline_data[:max_events]

        timeline_response = {
            "timeline": timeline_data,
            "metadata": {
                "total_events_shown": len(timeline_data),
                "unique_processes": len(processes),
                "unique_threads": len(threads),
                "time_range": {
                    "start": (
                        min(e["timestamp"] for e in timeline_data)
                        if timeline_data
                        else 0
                    ),
                    "end": (
                        max(e["timestamp"] for e in timeline_data)
                        if timeline_data
                        else 0
                    ),
                },
            },
        }

        return APIResponse.success(timeline_response)

    def handle_hotspots(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/time-trace/hotspots - Find performance hotspots"""
        parsed_data = self.get_parsed_data()

        event_durations = defaultdict(list)
        category_times = defaultdict(float)
        thread_times = defaultdict(float)

        for unit_name, unit_data in parsed_data.items():
            if FileType.TIME_TRACE in unit_data:
                for parsed_file in unit_data[FileType.TIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        for event in parsed_file.data:
                            if (
                                isinstance(event, TraceEvent)
                                and event.duration is not None
                            ):
                                event_durations[event.name].append(event.duration)
                                category_times[event.category] += event.duration

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

    def handle_categories(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/time-trace/categories - Analysis by event categories"""
        parsed_data = self.get_parsed_data()

        categories_data = defaultdict(
            lambda: {
                "events": [],
                "total_time": 0,
                "event_names": Counter(),
                "phases": Counter(),
            }
        )

        for unit_name, unit_data in parsed_data.items():
            if FileType.TIME_TRACE in unit_data:
                for parsed_file in unit_data[FileType.TIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        for event in parsed_file.data:
                            if isinstance(event, TraceEvent):
                                category = event.category or "uncategorized"

                                categories_data[category]["event_names"][
                                    event.name
                                ] += 1
                                categories_data[category]["phases"][event.phase] += 1

                                if event.duration is not None:
                                    categories_data[category][
                                        "total_time"
                                    ] += event.duration

                                # Keep sample events (limited)
                                if len(categories_data[category]["events"]) < 5:
                                    categories_data[category]["events"].append(
                                        {
                                            "name": event.name,
                                            "duration": event.duration,
                                            "timestamp": event.timestamp,
                                            "args": event.args if event.args else {},
                                        }
                                    )

        # Convert to response format
        result = {}
        for category, data in categories_data.items():
            result[category] = {
                "total_time": data["total_time"],
                "unique_event_names": len(data["event_names"]),
                "total_events": sum(data["event_names"].values()),
                "top_events": dict(data["event_names"].most_common(5)),
                "phase_distribution": dict(data["phases"]),
                "sample_events": data["events"],
            }

        # Sort by total time
        sorted_result = dict(
            sorted(result.items(), key=lambda x: x[1]["total_time"], reverse=True)
        )

        return APIResponse.success(
            {"categories": sorted_result, "total_categories": len(sorted_result)}
        )

    def handle_parallelism(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/time-trace/parallelism - Analyze parallelism and concurrency"""
        parsed_data = self.get_parsed_data()

        thread_activity = defaultdict(list)
        process_threads = defaultdict(set)
        concurrent_events = []

        for unit_name, unit_data in parsed_data.items():
            if FileType.TIME_TRACE in unit_data:
                for parsed_file in unit_data[FileType.TIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        for event in parsed_file.data:
                            if isinstance(event, TraceEvent):
                                if event.pid is not None and event.tid is not None:
                                    process_threads[event.pid].add(event.tid)

                                    if event.duration is not None:
                                        start_time = event.timestamp
                                        end_time = event.timestamp + event.duration

                                        thread_activity[event.tid].append(
                                            {
                                                "start": start_time,
                                                "end": end_time,
                                                "duration": event.duration,
                                                "event_name": event.name,
                                                "category": event.category,
                                            }
                                        )

        # Analyze thread utilization
        thread_utilization = {}
        for tid, activities in thread_activity.items():
            if activities:
                total_active_time = sum(a["duration"] for a in activities)
                activities.sort(key=lambda x: x["start"])

                time_span = activities[-1]["end"] - activities[0]["start"]
                utilization = (
                    (total_active_time / max(time_span, 1)) if time_span > 0 else 0
                )

                thread_utilization[tid] = {
                    "total_active_time": total_active_time,
                    "time_span": time_span,
                    "utilization_percentage": utilization * 100,
                    "activity_count": len(activities),
                    "average_activity_duration": total_active_time / len(activities),
                }

        # Find overlapping events (basic concurrency analysis)
        overlaps = 0
        sorted_activities = []
        for activities in thread_activity.values():
            sorted_activities.extend(activities)

        sorted_activities.sort(key=lambda x: x["start"])

        for i in range(len(sorted_activities) - 1):
            current = sorted_activities[i]
            next_activity = sorted_activities[i + 1]

            if current["end"] > next_activity["start"]:
                overlaps += 1

        parallelism_data = {
            "process_thread_mapping": {
                pid: len(threads) for pid, threads in process_threads.items()
            },
            "thread_utilization": dict(
                sorted(
                    thread_utilization.items(),
                    key=lambda x: x[1]["utilization_percentage"],
                    reverse=True,
                )
            ),
            "concurrency_metrics": {
                "total_threads": len(thread_activity),
                "overlapping_activities": overlaps,
                "max_threads_per_process": (
                    max(len(threads) for threads in process_threads.values())
                    if process_threads
                    else 0
                ),
            },
            "insights": self._generate_parallelism_insights(
                thread_utilization, process_threads
            ),
        }

        return APIResponse.success(parallelism_data)

    def _generate_parallelism_insights(
        self, thread_utilization: Dict, process_threads: Dict
    ) -> List[str]:
        """Generate insights about parallelism"""
        insights = []

        if thread_utilization:
            avg_utilization = sum(
                t["utilization_percentage"] for t in thread_utilization.values()
            ) / len(thread_utilization)

            if avg_utilization < 50:
                insights.append(
                    "Low thread utilization detected - potential for better parallelization"
                )
            elif avg_utilization > 90:
                insights.append("High thread utilization - good parallelization")

        total_threads = sum(len(threads) for threads in process_threads.values())
        if total_threads > 8:
            insights.append(
                f"High thread count ({total_threads}) - monitor for contention"
            )

        return insights

    def handle_flamegraph(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/time-trace/flamegraph - Get flamegraph data for time order view"""
        parsed_data = self.get_parsed_data()
        unit_name = query_params.get("unit", [None])[0]

        if unit_name and unit_name in parsed_data:
            unit_data = {unit_name: parsed_data[unit_name]}
        else:
            unit_data = parsed_data

        all_stacks = []

        for unit, data in unit_data.items():
            if FileType.TIME_TRACE in data:
                for parsed_file in data[FileType.TIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        from common.parsers.time_trace_parser import TimeTraceParser

                        parser = TimeTraceParser()
                        flamegraph_data = parser.get_flamegraph_data(parsed_file.data)

                        for sample in flamegraph_data.get("samples", []):
                            sample["unit"] = unit
                            all_stacks.append(sample)

        # Sort by timestamp for time order view
        all_stacks.sort(key=lambda x: x.get("timestamp", 0))

        return APIResponse.success(
            {"samples": all_stacks, "total_samples": len(all_stacks)}
        )

    def handle_sandwich(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/time-trace/sandwich - Get sandwich view data (aggregated by function)"""
        parsed_data = self.get_parsed_data()
        unit_name = query_params.get("unit", [None])[0]

        if unit_name and unit_name in parsed_data:
            unit_data = {unit_name: parsed_data[unit_name]}
        else:
            unit_data = parsed_data

        all_functions = []

        for unit, data in unit_data.items():
            if FileType.TIME_TRACE in data:
                for parsed_file in data[FileType.TIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        from common.parsers.time_trace_parser import TimeTraceParser

                        parser = TimeTraceParser()
                        sandwich_data = parser.get_sandwich_data(parsed_file.data)

                        for func in sandwich_data.get("functions", []):
                            func["unit"] = unit
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
                    "category": func.get("category", ""),
                    "units": [],
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

    def handle_runtime_comparison(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/time-trace/runtime-comparison - Compare time-trace vs runtime-trace"""
        parsed_data = self.get_parsed_data()

        time_trace_data = {}
        runtime_trace_data = {}

        # Collect both types of traces
        for unit_name, unit_data in parsed_data.items():
            if FileType.TIME_TRACE in unit_data:
                for parsed_file in unit_data[FileType.TIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        time_trace_data[unit_name] = parsed_file.data

            if FileType.RUNTIME_TRACE in unit_data:
                for parsed_file in unit_data[FileType.RUNTIME_TRACE]:
                    if isinstance(parsed_file.data, list):
                        runtime_trace_data[unit_name] = parsed_file.data

        comparison = {}

        for unit_name in set(time_trace_data.keys()) | set(runtime_trace_data.keys()):
            time_events = time_trace_data.get(unit_name, [])
            runtime_events = runtime_trace_data.get(unit_name, [])

            time_duration = sum(e.duration for e in time_events if e.duration)
            runtime_duration = sum(e.duration for e in runtime_events if e.duration)

            comparison[unit_name] = {
                "time_trace": {
                    "events": len(time_events),
                    "total_duration": time_duration,
                    "available": len(time_events) > 0,
                },
                "runtime_trace": {
                    "events": len(runtime_events),
                    "total_duration": runtime_duration,
                    "available": len(runtime_events) > 0,
                },
            }

        return APIResponse.success(
            {"comparison": comparison, "units": list(comparison.keys())}
        )
