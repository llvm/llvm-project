# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from typing import Dict, Any
import sys
from pathlib import Path
from ..base import APIResponse


class BaseSpecializedEndpoint:
    """Base class for specialized file-type endpoints"""

    def __init__(self, data_dir: str, collector):
        self.data_dir = data_dir
        self.collector = collector
        self._cache = {}

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
