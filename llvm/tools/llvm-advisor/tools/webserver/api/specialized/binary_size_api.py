# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import sys
from collections import defaultdict, Counter
from typing import Dict, Any
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
tools_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(tools_dir))

from common.models import FileType, BinarySize
from ..base import BaseEndpoint, APIResponse


class BinarySizeEndpoint(BaseEndpoint):
    """Specialized endpoints for binary size analysis"""

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
        """GET /api/binary-size/overview - Overall binary size statistics"""
        parsed_data = self.get_parsed_data()

        total_size = 0
        section_sizes = defaultdict(int)
        section_counts = Counter()
        size_distribution = []

        for unit_name, unit_data in parsed_data.items():
            if FileType.BINARY_SIZE in unit_data:
                for parsed_file in unit_data[FileType.BINARY_SIZE]:
                    if isinstance(parsed_file.data, list):
                        for size_entry in parsed_file.data:
                            if isinstance(size_entry, BinarySize):
                                total_size += size_entry.size
                                section_sizes[size_entry.section] += size_entry.size
                                section_counts[size_entry.section] += 1
                                size_distribution.append(size_entry.size)

        # Calculate size statistics
        size_stats = {}
        if size_distribution:
            size_distribution.sort()
            size_stats = {
                "total_size": total_size,
                "average_section_size": total_size / len(size_distribution),
                "median_section_size": size_distribution[len(size_distribution) // 2],
                "largest_section_size": max(size_distribution),
                "smallest_section_size": min(size_distribution),
                "total_sections": len(size_distribution),
            }

        overview_data = {
            "size_statistics": size_stats,
            "section_breakdown": dict(
                sorted(section_sizes.items(), key=lambda x: x[1], reverse=True)[:15]
            ),
            "section_counts": dict(section_counts.most_common(10)),
            "size_insights": self._generate_size_insights(section_sizes, total_size),
        }

        return APIResponse.success(overview_data)

    def handle_sections(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/binary-size/sections - Detailed analysis by sections"""
        parsed_data = self.get_parsed_data()

        sections_data = defaultdict(
            lambda: {
                "total_size": 0,
                "occurrences": 0,
                "units": set(),
                "size_distribution": [],
            }
        )

        for unit_name, unit_data in parsed_data.items():
            if FileType.BINARY_SIZE in unit_data:
                for parsed_file in unit_data[FileType.BINARY_SIZE]:
                    if isinstance(parsed_file.data, list):
                        for size_entry in parsed_file.data:
                            if isinstance(size_entry, BinarySize):
                                section = size_entry.section
                                sections_data[section]["total_size"] += size_entry.size
                                sections_data[section]["occurrences"] += 1
                                sections_data[section]["units"].add(unit_name)
                                sections_data[section]["size_distribution"].append(
                                    size_entry.size
                                )

        # Convert to detailed analysis
        result = {}
        for section, data in sections_data.items():
            sizes = data["size_distribution"]
            sizes.sort()

            result[section] = {
                "total_size": data["total_size"],
                "occurrences": data["occurrences"],
                "units_involved": len(data["units"]),
                "average_size": (
                    data["total_size"] / data["occurrences"]
                    if data["occurrences"] > 0
                    else 0
                ),
                "size_range": {
                    "min": min(sizes) if sizes else 0,
                    "max": max(sizes) if sizes else 0,
                    "median": sizes[len(sizes) // 2] if sizes else 0,
                },
                "section_type": self._classify_section_type(section),
                "optimization_potential": self._assess_optimization_potential(
                    section, data["total_size"]
                ),
            }

        # Sort by total size
        sorted_result = dict(
            sorted(result.items(), key=lambda x: x[1]["total_size"], reverse=True)
        )

        return APIResponse.success(
            {"sections": sorted_result, "total_unique_sections": len(sorted_result)}
        )

    def handle_optimization(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/binary-size/optimization - Size optimization opportunities"""
        parsed_data = self.get_parsed_data()

        large_sections = []
        redundant_sections = defaultdict(list)
        optimization_opportunities = []

        # Collect all size data
        all_sections = defaultdict(list)

        for unit_name, unit_data in parsed_data.items():
            if FileType.BINARY_SIZE in unit_data:
                for parsed_file in unit_data[FileType.BINARY_SIZE]:
                    if isinstance(parsed_file.data, list):
                        for size_entry in parsed_file.data:
                            if isinstance(size_entry, BinarySize):
                                all_sections[size_entry.section].append(
                                    {
                                        "size": size_entry.size,
                                        "unit": unit_name,
                                        "percentage": size_entry.percentage or 0,
                                    }
                                )

        # Find optimization opportunities
        total_binary_size = sum(
            sum(entries[0]["size"] for entries in all_sections.values())
        )

        for section, entries in all_sections.items():
            total_section_size = sum(entry["size"] for entry in entries)
            section_percentage = (total_section_size / max(total_binary_size, 1)) * 100

            # Large sections (>5% of total)
            if section_percentage > 5:
                large_sections.append(
                    {
                        "section": section,
                        "total_size": total_section_size,
                        "percentage": section_percentage,
                        "occurrences": len(entries),
                        "optimization_type": "large_section",
                    }
                )

            # Redundant sections (same name, multiple occurrences)
            if len(entries) > 1:
                redundant_sections[section] = {
                    "total_size": total_section_size,
                    "occurrences": len(entries),
                    "average_size": total_section_size / len(entries),
                    "units": [entry["unit"] for entry in entries],
                }

        # Generate specific optimization recommendations
        optimization_opportunities = self._generate_optimization_recommendations(
            large_sections, redundant_sections, total_binary_size
        )

        optimization_data = {
            "large_sections": sorted(
                large_sections, key=lambda x: x["total_size"], reverse=True
            ),
            "redundant_sections": dict(redundant_sections),
            "optimization_opportunities": optimization_opportunities,
            "potential_savings": self._calculate_potential_savings(
                large_sections, redundant_sections
            ),
            "binary_size_breakdown": {
                "total_size": total_binary_size,
                "largest_contributors": [
                    s["section"]
                    for s in sorted(
                        large_sections, key=lambda x: x["total_size"], reverse=True
                    )[:5]
                ],
            },
        }

        return APIResponse.success(optimization_data)

    def handle_comparison(
        self, path_parts: list, query_params: Dict[str, list]
    ) -> Dict[str, Any]:
        """GET /api/binary-size/comparison - Compare sizes across compilation units"""
        parsed_data = self.get_parsed_data()

        unit_sizes = {}
        section_comparison = defaultdict(dict)

        for unit_name, unit_data in parsed_data.items():
            if FileType.BINARY_SIZE in unit_data:
                unit_total = 0
                unit_sections = {}

                for parsed_file in unit_data[FileType.BINARY_SIZE]:
                    if isinstance(parsed_file.data, list):
                        for size_entry in parsed_file.data:
                            if isinstance(size_entry, BinarySize):
                                unit_total += size_entry.size
                                unit_sections[size_entry.section] = size_entry.size
                                section_comparison[size_entry.section][
                                    unit_name
                                ] = size_entry.size

                unit_sizes[unit_name] = {
                    "total_size": unit_total,
                    "section_count": len(unit_sections),
                    "sections": unit_sections,
                    "largest_section": (
                        max(unit_sections.items(), key=lambda x: x[1])
                        if unit_sections
                        else ("", 0)
                    ),
                }

        # Calculate comparison metrics
        if unit_sizes:
            sizes = [data["total_size"] for data in unit_sizes.values()]
            avg_size = sum(sizes) / len(sizes)

            comparison_metrics = {
                "average_unit_size": avg_size,
                "largest_unit": max(
                    unit_sizes.items(), key=lambda x: x[1]["total_size"]
                ),
                "smallest_unit": min(
                    unit_sizes.items(), key=lambda x: x[1]["total_size"]
                ),
                "size_variance": self._calculate_variance(sizes),
                "units_above_average": [
                    name
                    for name, data in unit_sizes.items()
                    if data["total_size"] > avg_size
                ],
            }
        else:
            comparison_metrics = {}

        comparison_data = {
            "unit_comparison": dict(
                sorted(
                    unit_sizes.items(), key=lambda x: x[1]["total_size"], reverse=True
                )
            ),
            "section_comparison": dict(section_comparison),
            "comparison_metrics": comparison_metrics,
            "insights": self._generate_comparison_insights(
                unit_sizes, comparison_metrics
            ),
        }

        return APIResponse.success(comparison_data)

    def _generate_size_insights(
        self, section_sizes: Dict[str, int], total_size: int
    ) -> list:
        """Generate insights about binary size"""
        insights = []

        if section_sizes:
            largest_section = max(section_sizes.items(), key=lambda x: x[1])
            largest_percentage = (largest_section[1] / max(total_size, 1)) * 100

            if largest_percentage > 50:
                insights.append(
                    f"'{largest_section[0]}' dominates binary size ({largest_percentage:.1f}%)"
                )

            text_sections = [k for k in section_sizes.keys() if "text" in k.lower()]
            if text_sections:
                text_size = sum(section_sizes[k] for k in text_sections)
                text_percentage = (text_size / max(total_size, 1)) * 100
                insights.append(
                    f"Code sections account for {text_percentage:.1f}% of binary size"
                )

        return insights

    def _classify_section_type(self, section_name: str) -> str:
        """Classify section by type"""
        section_lower = section_name.lower()

        if "text" in section_lower or "code" in section_lower:
            return "code"
        elif "data" in section_lower:
            return "data"
        elif "bss" in section_lower:
            return "uninitialized_data"
        elif "rodata" in section_lower or "const" in section_lower:
            return "read_only_data"
        elif "debug" in section_lower:
            return "debug_info"
        else:
            return "other"

    def _assess_optimization_potential(self, section: str, size: int) -> str:
        """Assess optimization potential for a section"""
        if size > 1024 * 1024:  # >1MB
            return "high"
        elif size > 256 * 1024:  # >256KB
            return "medium"
        else:
            return "low"

    def _generate_optimization_recommendations(
        self, large_sections: list, redundant_sections: dict, total_size: int
    ) -> list:
        """Generate specific optimization recommendations"""
        recommendations = []

        if large_sections:
            recommendations.append(
                {
                    "type": "large_sections",
                    "priority": "high",
                    "description": f"Consider optimizing {len(large_sections)} large sections",
                    "sections": [s["section"] for s in large_sections[:3]],
                }
            )

        if redundant_sections:
            recommendations.append(
                {
                    "type": "redundant_sections",
                    "priority": "medium",
                    "description": f"Found {len(redundant_sections)} potentially redundant sections",
                    "sections": list(redundant_sections.keys())[:3],
                }
            )

        if total_size > 10 * 1024 * 1024:  # >10MB
            recommendations.append(
                {
                    "type": "overall_size",
                    "priority": "medium",
                    "description": "Binary size is large - consider link-time optimization",
                }
            )

        return recommendations

    def _calculate_potential_savings(
        self, large_sections: list, redundant_sections: dict
    ) -> Dict[str, Any]:
        """Calculate potential size savings"""
        large_section_savings = sum(
            s["total_size"] * 0.1 for s in large_sections
        )  # Assume 10% reduction
        redundant_savings = sum(
            data["total_size"] * 0.2 for data in redundant_sections.values()
        )  # Assume 20% reduction

        return {
            "from_large_sections": int(large_section_savings),
            "from_redundant_sections": int(redundant_savings),
            "total_potential": int(large_section_savings + redundant_savings),
        }

    def _calculate_variance(self, values: list) -> float:
        """Calculate variance of values"""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _generate_comparison_insights(self, unit_sizes: dict, metrics: dict) -> list:
        """Generate insights from unit comparison"""
        insights = []

        if (
            metrics.get("size_variance", 0)
            > (metrics.get("average_unit_size", 0) * 0.25) ** 2
        ):
            insights.append(
                "High size variance between units - investigate inconsistencies"
            )

        if metrics.get("units_above_average"):
            insights.append(
                f"{len(metrics['units_above_average'])} units are above average size"
            )

        return insights
