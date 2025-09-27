# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import sys
from collections import defaultdict
from typing import Dict, Any, List
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
tools_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(tools_dir))

from common.models import FileType
from ..base import BaseEndpoint, APIResponse


class CompilationPhasesEndpoint(BaseEndpoint):
    """Specialized endpoints for compilation phases timing analysis"""

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
        """GET /api/compilation-phases/overview - Overall compilation timing statistics"""
        parsed_data = self.get_parsed_data()

        total_time = 0
        phase_times = defaultdict(list)
        unit_times = {}

        for unit_name, unit_data in parsed_data.items():
            if FileType.COMPILATION_PHASES in unit_data:
                unit_total = 0
                for parsed_file in unit_data[FileType.COMPILATION_PHASES]:
                    if (
                        isinstance(parsed_file.data, dict)
                        and "phases" in parsed_file.data
                    ):
                        for phase in parsed_file.data["phases"]:
                            if phase.get("duration") is not None:
                                duration = phase["duration"]
                                phase_name = phase["name"]

                                phase_times[phase_name].append(duration)
                                unit_total += duration
                                total_time += duration

                if unit_total > 0:
                    unit_times[unit_name] = unit_total

        # Calculate phase statistics
        phase_stats = {}
        for phase_name, times in phase_times.items():
            if times:
                phase_stats[phase_name] = {
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "max_time": max(times),
                    "min_time": min(times),
                    "occurrences": len(times),
                    "percentage": (sum(times) / max(total_time, 1)) * 100,
                }

        # Sort by total time
        sorted_phases = dict(
            sorted(phase_stats.items(), key=lambda x: x[1]["total_time"], reverse=True)
        )

        overview_data = {
            "totals": {
                "compilation_time": total_time,
                "unique_phases": len(phase_stats),
                "compilation_units": len(unit_times),
            },
            "phase_breakdown": sorted_phases,
            "unit_times": dict(
                sorted(unit_times.items(), key=lambda x: x[1], reverse=True)
            ),
            "top_time_consumers": dict(list(sorted_phases.items())[:5]),
        }

        return APIResponse.success(overview_data)

    def handle_phases(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/compilation-phases/phases - Detailed analysis by individual phases"""
        parsed_data = self.get_parsed_data()

        phases_data = defaultdict(lambda: {"times": [], "units": set(), "files": []})

        for unit_name, unit_data in parsed_data.items():
            if FileType.COMPILATION_PHASES in unit_data:
                for parsed_file in unit_data[FileType.COMPILATION_PHASES]:
                    if (
                        isinstance(parsed_file.data, dict)
                        and "phases" in parsed_file.data
                    ):
                        for phase in parsed_file.data["phases"]:
                            if phase.get("duration") is not None:
                                phase_name = phase["name"]
                                duration = phase["duration"]

                                phases_data[phase_name]["times"].append(duration)
                                phases_data[phase_name]["units"].add(unit_name)
                                phases_data[phase_name]["files"].append(
                                    {
                                        "unit": unit_name,
                                        "file": parsed_file.file_path,
                                        "duration": duration,
                                        "info": phase.get("info", ""),
                                    }
                                )

        # Convert to detailed statistics
        result = {}
        for phase_name, data in phases_data.items():
            times = data["times"]
            if times:
                result[phase_name] = {
                    "statistics": {
                        "count": len(times),
                        "total_time": sum(times),
                        "average_time": sum(times) / len(times),
                        "median_time": sorted(times)[len(times) // 2],
                        "max_time": max(times),
                        "min_time": min(times),
                        "std_deviation": self._calculate_std_dev(times),
                    },
                    "distribution": {
                        "units_involved": len(data["units"]),
                        "files_processed": len(data["files"]),
                    },
                    "performance_insights": self._analyze_phase_performance(times),
                    "slowest_instances": sorted(
                        data["files"], key=lambda x: x["duration"], reverse=True
                    )[:3],
                }

        # Sort by total time
        sorted_result = dict(
            sorted(
                result.items(),
                key=lambda x: x[1]["statistics"]["total_time"],
                reverse=True,
            )
        )

        return APIResponse.success(
            {"phases": sorted_result, "total_phases_analyzed": len(sorted_result)}
        )

    def handle_bottlenecks(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/compilation-phases/bottlenecks - Identify compilation bottlenecks"""
        parsed_data = self.get_parsed_data()

        all_phase_times = []
        phase_outliers = defaultdict(list)
        unit_bottlenecks = {}

        for unit_name, unit_data in parsed_data.items():
            if FileType.COMPILATION_PHASES in unit_data:
                unit_phases = []

                for parsed_file in unit_data[FileType.COMPILATION_PHASES]:
                    if (
                        isinstance(parsed_file.data, dict)
                        and "phases" in parsed_file.data
                    ):
                        for phase in parsed_file.data["phases"]:
                            if phase.get("duration") is not None:
                                duration = phase["duration"]
                                phase_name = phase["name"]

                                all_phase_times.append(duration)
                                unit_phases.append(
                                    {
                                        "name": phase_name,
                                        "duration": duration,
                                        "file": parsed_file.file_path,
                                    }
                                )

                if unit_phases:
                    # Find bottlenecks in this unit
                    unit_phases.sort(key=lambda x: x["duration"], reverse=True)
                    total_time = sum(p["duration"] for p in unit_phases)

                    unit_bottlenecks[unit_name] = {
                        "total_time": total_time,
                        "slowest_phase": unit_phases[0] if unit_phases else None,
                        "top_3_phases": unit_phases[:3],
                        "phase_distribution": self._calculate_phase_distribution(
                            unit_phases
                        ),
                    }

        # Calculate global thresholds for outlier detection
        if all_phase_times:
            all_phase_times.sort()
            p95_threshold = all_phase_times[int(len(all_phase_times) * 0.95)]
            p99_threshold = all_phase_times[int(len(all_phase_times) * 0.99)]

            # Find outliers
            for unit_name, unit_data in parsed_data.items():
                if FileType.COMPILATION_PHASES in unit_data:
                    for parsed_file in unit_data[FileType.COMPILATION_PHASES]:
                        if (
                            isinstance(parsed_file.data, dict)
                            and "phases" in parsed_file.data
                        ):
                            for phase in parsed_file.data["phases"]:
                                if phase.get("duration") is not None:
                                    duration = phase["duration"]

                                    if duration >= p99_threshold:
                                        phase_outliers["p99"].append(
                                            {
                                                "unit": unit_name,
                                                "phase": phase["name"],
                                                "duration": duration,
                                                "file": parsed_file.file_path,
                                            }
                                        )
                                    elif duration >= p95_threshold:
                                        phase_outliers["p95"].append(
                                            {
                                                "unit": unit_name,
                                                "phase": phase["name"],
                                                "duration": duration,
                                                "file": parsed_file.file_path,
                                            }
                                        )

        bottlenecks_data = {
            "global_thresholds": {
                "p95_threshold": p95_threshold if all_phase_times else 0,
                "p99_threshold": p99_threshold if all_phase_times else 0,
            },
            "outliers": dict(phase_outliers),
            "unit_bottlenecks": dict(
                sorted(
                    unit_bottlenecks.items(),
                    key=lambda x: x[1]["total_time"],
                    reverse=True,
                )
            ),
            "recommendations": self._generate_bottleneck_recommendations(
                unit_bottlenecks, phase_outliers
            ),
        }

        return APIResponse.success(bottlenecks_data)

    def handle_trends(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/compilation-phases/trends - Compilation time trends and patterns"""
        parsed_data = self.get_parsed_data()

        # For this endpoint, we'll analyze patterns across units
        # This needs more work to identify trends over time
        phase_consistency = defaultdict(list)
        unit_patterns = {}

        for unit_name, unit_data in parsed_data.items():
            if FileType.COMPILATION_PHASES in unit_data:
                unit_phase_times = defaultdict(list)

                for parsed_file in unit_data[FileType.COMPILATION_PHASES]:
                    if (
                        isinstance(parsed_file.data, dict)
                        and "phases" in parsed_file.data
                    ):
                        for phase in parsed_file.data["phases"]:
                            if phase.get("duration") is not None:
                                phase_name = phase["name"]
                                duration = phase["duration"]

                                unit_phase_times[phase_name].append(duration)
                                phase_consistency[phase_name].append(duration)

                # Analyze patterns for this unit
                if unit_phase_times:
                    unit_patterns[unit_name] = self._analyze_unit_patterns(
                        unit_phase_times
                    )

        # Calculate consistency metrics
        consistency_metrics = {}
        for phase_name, times in phase_consistency.items():
            if len(times) > 1:
                avg_time = sum(times) / len(times)
                variance = sum((t - avg_time) ** 2 for t in times) / len(times)
                coefficient_of_variation = (
                    (variance**0.5) / avg_time if avg_time > 0 else 0
                )

                consistency_metrics[phase_name] = {
                    "coefficient_of_variation": coefficient_of_variation,
                    "consistency_rating": (
                        "high"
                        if coefficient_of_variation < 0.2
                        else "medium"
                        if coefficient_of_variation < 0.5
                        else "low"
                    ),
                    "sample_size": len(times),
                }

        trends_data = {
            "phase_consistency": consistency_metrics,
            "unit_patterns": unit_patterns,
            "insights": self._generate_trend_insights(
                consistency_metrics, unit_patterns
            ),
        }

        return APIResponse.success(trends_data)

    def handle_bindings(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/compilation-phases/bindings - Compilation tool bindings from -ccc-print-bindings"""
        parsed_data = self.get_parsed_data()

        all_bindings = []
        tool_counts = defaultdict(int)
        target_counts = defaultdict(int)
        unit_summaries = {}

        for unit_name, unit_data in parsed_data.items():
            if FileType.COMPILATION_PHASES in unit_data:
                unit_bindings = []
                unit_tool_counts = defaultdict(int)

                for parsed_file in unit_data[FileType.COMPILATION_PHASES]:
                    if (
                        isinstance(parsed_file.data, dict)
                        and "bindings" in parsed_file.data
                    ):
                        bindings = parsed_file.data["bindings"]
                        unit_bindings.extend(bindings)
                        all_bindings.extend(bindings)

                        for binding in bindings:
                            tool = binding.get("tool", "unknown")
                            target = binding.get("target", "unknown")

                            tool_counts[tool] += 1
                            target_counts[target] += 1
                            unit_tool_counts[tool] += 1

                if unit_bindings:
                    unit_summaries[unit_name] = {
                        "total_bindings": len(unit_bindings),
                        "unique_tools": len(unit_tool_counts),
                        "tool_counts": dict(unit_tool_counts),
                        "bindings": unit_bindings,
                    }

        # Sort tools by usage count
        sorted_tool_counts = dict(
            sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        )

        sorted_target_counts = dict(
            sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
        )

        bindings_data = {
            "summary": {
                "total_bindings": len(all_bindings),
                "unique_tools": len(tool_counts),
                "unique_targets": len(target_counts),
                "compilation_units": len(unit_summaries),
            },
            "tool_counts": sorted_tool_counts,
            "target_counts": sorted_target_counts,
            "unit_summaries": unit_summaries,
            "all_bindings": all_bindings,
        }

        return APIResponse.success(bindings_data)

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def _analyze_phase_performance(self, times: List[float]) -> Dict[str, Any]:
        """Analyze performance characteristics of a phase"""
        if not times:
            return {}

        avg_time = sum(times) / len(times)
        max_time = max(times)

        return {
            "variability": "high" if max_time > avg_time * 2 else "low",
            "performance_rating": "fast" if avg_time < 100 else "slow",
            "consistency": "consistent" if max_time < avg_time * 1.5 else "variable",
        }

    def _calculate_phase_distribution(
        self, phases: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate time distribution across phases"""
        total_time = sum(p["duration"] for p in phases)
        if total_time == 0:
            return {}

        distribution = {}
        for phase in phases:
            phase_name = phase["name"]
            percentage = (phase["duration"] / total_time) * 100
            if phase_name not in distribution:
                distribution[phase_name] = 0
            distribution[phase_name] += percentage

        return distribution

    def _generate_bottleneck_recommendations(
        self, unit_bottlenecks: Dict, outliers: Dict
    ) -> List[str]:
        """Generate recommendations for addressing bottlenecks"""
        recommendations = []

        if outliers.get("p99"):
            recommendations.append(
                "Consider investigating phases with >99th percentile timing"
            )

        slow_units = [
            name
            for name, data in unit_bottlenecks.items()
            if data.get("total_time", 0) > 1000
        ]  # > 1 second

        if slow_units:
            recommendations.append(
                f"Units with high compile times: {', '.join(slow_units[:3])}"
            )

        return recommendations

    def _analyze_unit_patterns(self, unit_phase_times: Dict) -> Dict[str, Any]:
        """Analyze compilation patterns for a unit"""
        total_phases = len(unit_phase_times)
        dominant_phase = (
            max(unit_phase_times.items(), key=lambda x: sum(x[1]))
            if unit_phase_times
            else None
        )

        return {
            "total_unique_phases": total_phases,
            "dominant_phase": dominant_phase[0] if dominant_phase else None,
            "phase_count": sum(len(times) for times in unit_phase_times.values()),
        }

    def _generate_trend_insights(
        self, consistency_metrics: Dict, unit_patterns: Dict
    ) -> List[str]:
        """Generate insights about compilation trends"""
        insights = []

        consistent_phases = [
            name
            for name, metrics in consistency_metrics.items()
            if metrics["consistency_rating"] == "high"
        ]

        if consistent_phases:
            insights.append(
                f"Highly consistent phases: {', '.join(consistent_phases[:3])}"
            )

        variable_phases = [
            name
            for name, metrics in consistency_metrics.items()
            if metrics["consistency_rating"] == "low"
        ]

        if variable_phases:
            insights.append(
                f"Variable phases needing attention: {', '.join(variable_phases[:3])}"
            )

        return insights
