# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from typing import Dict, Any
from .base import BaseEndpoint, APIResponse


class SummaryEndpoint(BaseEndpoint):
    """GET /api/summary - Overall statistics summary across all compilation units"""

    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        parsed_data = self.get_parsed_data()
        stats = self.collector.get_summary_statistics(parsed_data)

        # Enhance with additional summary metrics
        enhanced_stats = {
            **stats,
            "status": "success" if stats["errors"] == 0 else "partial_errors",
            "success_rate": (
                (stats["total_files"] - stats["errors"]) / stats["total_files"] * 100
                if stats["total_files"] > 0
                else 0
            ),
        }

        return APIResponse.success(enhanced_stats)
