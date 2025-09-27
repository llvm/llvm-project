# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import os
from typing import Dict, Any
from .base import BaseEndpoint, APIResponse


class HealthEndpoint(BaseEndpoint):
    """GET /api/health - System health and data directory status"""

    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        data_dir_exists = os.path.exists(self.data_dir)

        # Count units and files if data directory exists
        unit_count = 0
        file_count = 0

        if data_dir_exists:
            try:
                units = self.get_compilation_units()
                unit_count = len(units)
                file_count = sum(
                    sum(len(files) for files in unit.artifacts.values())
                    for unit in units
                )
            except Exception:
                pass

        health_data = {
            "status": "healthy" if data_dir_exists else "no_data",
            "data_dir": self.data_dir,
            "data_dir_exists": data_dir_exists,
            "compilation_units": unit_count,
            "total_files": file_count,
        }

        return APIResponse.success(health_data)
