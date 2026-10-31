# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark variables and macros for building libc++ and libc++abi.

Variables provide base line information for how to build libc++ and libc++abi
source files. These can be used to generate non-Bazel builds of the library.

TODO: Add macros that provide a convenient way to construct Bazel `cc_library`
rules for libc++ and libc++abi.

TODO: Add either sufficient usage in the macros, or add a how-to example here in
the documentation so the use of these variables is more clear.
"""

_libcxx_base_copts = [
    "-std=c++26",
    "-O3",
    "-fPIC",
    "-fvisibility-inlines-hidden",
    "-fvisibility=hidden",
    "-nostdinc++",
]

_libcxx_defines = [
    "-D_LIBCPP_BUILDING_LIBRARY",
    "-D_LIBCPP_REMOVE_TRANSITIVE_INCLUDES",
]

_libcxxabi_defines = [
    "-DLIBCXX_BUILDING_LIBCXXABI",
]

libcxx_copts = _libcxx_base_copts + _libcxx_defines
libcxxabi_copts = _libcxx_base_copts + _libcxxabi_defines
libcxx_and_abi_copts = _libcxx_base_copts + _libcxx_defines + _libcxxabi_defines
