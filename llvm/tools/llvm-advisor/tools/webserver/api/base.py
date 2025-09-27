# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseEndpoint(ABC):
    def __init__(self, data_dir: str, collector):
        self.data_dir = data_dir
        self.collector = collector
        self._cache = {}

    @abstractmethod
    def handle(self, path_parts: list, query_params: Dict[str, list]) -> Dict[str, Any]:
        pass

    def get_compilation_units(self):
        if "units" not in self._cache:
            self._cache["units"] = self.collector.discover_compilation_units(
                self.data_dir
            )
        return self._cache["units"]

    def get_parsed_data(self):
        if "parsed_data" not in self._cache:
            self._cache["parsed_data"] = self.collector.parse_all_units(self.data_dir)
        return self._cache["parsed_data"]

    def clear_cache(self):
        self._cache.clear()


class APIResponse:
    @staticmethod
    def success(data: Any, status: int = 200) -> Dict[str, Any]:
        return {"success": True, "data": data, "status": status}

    @staticmethod
    def error(message: str, status: int = 400) -> Dict[str, Any]:
        return {"success": False, "error": message, "status": status}

    @staticmethod
    def not_found(resource: str) -> Dict[str, Any]:
        return APIResponse.error(f"{resource} not found", 404)

    @staticmethod
    def invalid_request(message: str) -> Dict[str, Any]:
        return APIResponse.error(f"Invalid request: {message}", 400)
