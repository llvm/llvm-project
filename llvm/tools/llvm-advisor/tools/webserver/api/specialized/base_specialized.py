# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

from typing import Optional


class BaseSpecializedEndpoint:
    """Base class for specialized file-type endpoints"""

    def __init__(self, data_dir: str, collector, data_store=None):
        self.data_dir = data_dir
        self.collector = collector
        self.data_store = data_store
        self._cache = {}

    def get_compilation_units(self):
        if self.data_store is not None:
            return self.data_store.get_compilation_units()

        if "units" not in self._cache:
            self._cache["units"] = self.collector.discover_compilation_units(
                self.data_dir
            )
        return self._cache["units"]

    def get_parsed_data(self, unit_name: Optional[str] = None):
        if self.data_store is not None:
            return self.data_store.get_parsed_data(unit_name=unit_name)

        if "parsed_data" not in self._cache:
            self._cache["parsed_data"] = self.collector.parse_all_units(self.data_dir)
        if unit_name is None:
            return self._cache["parsed_data"]

        if unit_name in self._cache["parsed_data"]:
            return {unit_name: self._cache["parsed_data"][unit_name]}
        return {}

    def clear_cache(self):
        if self.data_store is not None:
            self.data_store.clear_cache()
            return
        self._cache.clear()
