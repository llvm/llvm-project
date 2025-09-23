# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (c) 2025.

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os
import re

__all__ = ["dynamic_metadata"]


def __dir__() -> list[str]:
    return __all__


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
    _project: dict[str, object] = None,
) -> str:
    if field != "version":
        msg = "Only the 'version' field is supported"
        raise ValueError(msg)

    if settings:
        msg = "No inline configuration is supported"
        raise ValueError(msg)

    now = datetime.now()
    llvm_datetime = os.environ.get(
        "DATETIME", f"{now.year}{now.month:02}{now.day:02}{now.hour:02}"
    )

    llvm_src_root = Path(__file__).parent.parent.parent.parent.parent
    cmake_version_path = llvm_src_root / "cmake/Modules/LLVMVersion.cmake"
    if not cmake_version_path.exists():
        cmake_version_path = llvm_src_root / "llvm/CMakeLists.txt"
    if not cmake_version_path.exists():
        return llvm_datetime
    cmake_txt = open(cmake_version_path).read()
    llvm_version = []
    for v in ["LLVM_VERSION_MAJOR", "LLVM_VERSION_MINOR", "LLVM_VERSION_PATCH"]:
        vn = re.findall(rf"set\({v} (\d+)\)", cmake_txt)
        assert vn, f"couldn't find {v} in cmake txt"
        llvm_version.append(vn[0])

    return f"{llvm_version[0]}.{llvm_version[1]}.{llvm_version[2]}.{llvm_datetime}"
