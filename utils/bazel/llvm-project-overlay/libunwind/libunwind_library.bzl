# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark variables and macros for building libunwind.

Variables provide baseline information for how to build libunwind source files.
These can be used to generate non-Bazel builds of the library.

TODO: Add macros that provide a convenient way to construct Bazel `cc_library`
rules for libunwind.
"""

# TODO: Should libunwind use `-fvisibility-inlines-hidden` and
# `-fvisibility=hidden`, similar to libc++?
libunwind_copts = [
    "-D_LIBUNWIND_IS_NATIVE_ONLY",
    "-O3",
    "-fPIC",
    "-fno-exceptions",
    "-fno-rtti",
    "-funwind-tables",
    "-nostdinc++",
]
