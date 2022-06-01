# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc starlark rules for building individual functions."""

load(":platforms.bzl", "PLATFORM_CPU_ARM64", "PLATFORM_CPU_X86_64")
load("@bazel_skylib//lib:selects.bzl", "selects")

LIBC_ROOT_TARGET = ":libc_root"
INTERNAL_SUFFIX = ".__internal__"

def libc_function(name, srcs, deps = None, copts = None, **kwargs):
    """Add target for a libc function.

    The libc function is eventually available as a cc_library target by name
    "name". LLVM libc implementations of libc functions are in C++. So, this
    rule internally generates a C wrapper for the C++ implementation and adds
    it to the source list of the cc_library. This way, the C++ implementation
    and the C wrapper are both available in the cc_library.

    Args:
      name: Target name. It is normally the name of the function this target is
            for.
      srcs: The .cpp files which contain the function implementation.
      deps: The list of target dependencies if any.
      copts: The list of options to add to the C++ compilation command.
      **kwargs: Other attributes relevant for a cc_library. For example, deps.
    """
    deps = deps or []
    deps.append(LIBC_ROOT_TARGET)
    copts = copts or []
    copts.append("-O3")

    # We compile the code twice, the first target is suffixed with ".__internal__" and contains the
    # C++ functions in the "__llvm_libc" namespace. This allows us to test the function in the
    # presence of another libc.
    native.cc_library(
        name = name + INTERNAL_SUFFIX,
        srcs = srcs,
        deps = deps,
        copts = copts,
        linkstatic = 1,
        **kwargs
    )

    # This second target is the llvm libc C function.
    copts.append("-DLLVM_LIBC_PUBLIC_PACKAGING")
    native.cc_library(
        name = name,
        srcs = srcs,
        deps = deps,
        copts = copts,
        linkstatic = 1,
        **kwargs
    )

def libc_math_function(
        name,
        specializations = None,
        additional_deps = None):
    """Add a target for a math function.

    Args:
      name: The name of the function.
      specializations: List of machine specializations available for this
                       function. Possible specializations are "generic",
                       "aarch64" and "x86_64".
      additional_deps: Other deps like helper cc_library targes used by the
                       math function.
    """
    additional_deps = additional_deps or []
    specializations = specializations or ["generic"]
    select_map = {}
    if "generic" in specializations:
        select_map["//conditions:default"] = ["src/math/generic/" + name + ".cpp"]
    if "aarch64" in specializations:
        select_map[PLATFORM_CPU_ARM64] = ["src/math/aarch64/" + name + ".cpp"]
    if "x86_64" in specializations:
        select_map[PLATFORM_CPU_X86_64] = ["src/math/x86_64/" + name + ".cpp"]
    libc_function(
        name = name,
        srcs = selects.with_or(select_map),
        hdrs = ["src/math/" + name + ".h"],
        deps = [":__support_common", ":__support_fputil"] + additional_deps,
    )
