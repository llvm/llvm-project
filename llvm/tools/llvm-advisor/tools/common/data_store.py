# ===----------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//

import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from .models import CompilationUnit, FileType, ParsedFile


class AdvisorDataStore:
    """Shared, refreshable cache for discovered units and parsed artifacts."""

    def __init__(
        self, data_dir: str, collector, refresh_interval_seconds: float = 2.0
    ):
        self.data_dir = data_dir
        self.collector = collector
        self.refresh_interval_seconds = max(refresh_interval_seconds, 0.25)

        self._lock = threading.RLock()
        self._last_refresh_monotonic = 0.0
        self._fingerprint = None

        self._units: List[CompilationUnit] = []
        self._units_by_name: Dict[str, CompilationUnit] = {}
        self._parsed_units: Dict[str, Dict[FileType, List[ParsedFile]]] = {}

    def clear_cache(self):
        with self._lock:
            self._last_refresh_monotonic = 0.0
            self._fingerprint = None
            self._units = []
            self._units_by_name = {}
            self._parsed_units = {}

    def get_compilation_units(self) -> List[CompilationUnit]:
        with self._lock:
            self._refresh_if_needed()
            return list(self._units)

    def get_parsed_data(
        self, unit_name: Optional[str] = None
    ) -> Dict[str, Dict[FileType, List[ParsedFile]]]:
        with self._lock:
            self._refresh_if_needed()

            if unit_name is not None:
                self._ensure_unit_parsed(unit_name)
                unit_data = self._parsed_units.get(unit_name)
                if unit_data:
                    return {unit_name: unit_data}
                return {}

            for unit in self._units:
                self._ensure_unit_parsed(unit.name)

            return {name: data for name, data in self._parsed_units.items() if data}

    def _refresh_if_needed(self):
        now = time.monotonic()
        if (
            self._fingerprint is not None
            and now - self._last_refresh_monotonic < self.refresh_interval_seconds
        ):
            return

        current_fingerprint = self._compute_fingerprint()
        self._last_refresh_monotonic = now

        if current_fingerprint == self._fingerprint:
            return

        self._fingerprint = current_fingerprint
        self._units = self.collector.discover_compilation_units(self.data_dir)
        self._units_by_name = {unit.name: unit for unit in self._units}
        self._parsed_units = {}

    def _ensure_unit_parsed(self, unit_name: str):
        if unit_name in self._parsed_units:
            return

        unit = self._units_by_name.get(unit_name)
        if unit is None:
            return

        self._parsed_units[unit_name] = self.collector.parse_compilation_unit(unit)

    def _compute_fingerprint(self) -> Tuple[int, int, int, int]:
        if not os.path.exists(self.data_dir):
            return (0, 0, 0, 0)

        file_count = 0
        dir_count = 0
        total_size = 0
        latest_mtime_ns = 0

        for root, _dirs, files in os.walk(self.data_dir):
            dir_count += 1

            try:
                root_stat = os.stat(root)
                latest_mtime_ns = max(latest_mtime_ns, root_stat.st_mtime_ns)
            except OSError:
                continue

            for name in files:
                path = os.path.join(root, name)
                try:
                    stat_info = os.stat(path)
                except OSError:
                    continue

                file_count += 1
                total_size += stat_info.st_size
                latest_mtime_ns = max(latest_mtime_ns, stat_info.st_mtime_ns)

        return (file_count, dir_count, total_size, latest_mtime_ns)
