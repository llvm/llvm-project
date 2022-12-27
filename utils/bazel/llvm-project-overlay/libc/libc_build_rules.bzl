# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc starlark rules for building individual functions."""

load(":platforms.bzl", "PLATFORM_CPU_ARM64", "PLATFORM_CPU_X86_64")
load("@bazel_skylib//lib:selects.bzl", "selects")

LIBC_ROOT_TARGET = ":libc_root"
INTERNAL_SUFFIX = ".__internal__"

def _libc_library(name, copts = None, **kwargs):
    """Internal macro to serve as a base for all other libc library rules.

    Args:
      name: Target name.
      copts: The special compiler options for the target.
      **kwargs: All other attributes relevant for the cc_library rule.
    """
    copts = copts or []

    # We want all libc sources to be compiled with "hidden" visibility.
    # The public symbols will be given "default" visibility explicitly.
    # See src/__support/common.h for more information.
    copts.append("-fvisibility=hidden")
    native.cc_library(
        name = name,
        copts = copts,
        linkstatic = 1,
        **kwargs
    )

# A convenience var which should be used to list all libc support libraries.
# Any library which does not define a public function should be listed with
# libc_support_library.
libc_support_library = _libc_library

def libc_function(
        name,
        srcs,
        weak = False,
        deps = None,
        copts = None,
        local_defines = None,
        **kwargs):
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
      weak: Make the symbol corresponding to the libc function "weak".
      deps: The list of target dependencies if any.
      copts: The list of options to add to the C++ compilation command.
      local_defines: The preprocessor defines which will be prepended with -D
                     and passed to the compile command of this target but not
                     its deps.
      **kwargs: Other attributes relevant for a cc_library. For example, deps.
    """
    deps = deps or []
    deps.append(LIBC_ROOT_TARGET)
    copts = copts or []
    copts.append("-O3")
    copts.append("-fno-builtin")
    copts.append("-fno-lax-vector-conversions")

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

    func_attrs = ["__attribute__((visibility(\"default\")))"]
    if weak:
        func_attrs.append("__attribute__((weak))")
    local_defines = local_defines or ["LLVM_LIBC_PUBLIC_PACKAGING"]
    local_defines.append("LLVM_LIBC_FUNCTION_ATTR='%s'" % " ".join(func_attrs))
    _libc_library(
        name = name,
        srcs = srcs,
        deps = deps,
        copts = copts,
        local_defines = local_defines,
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

    #TODO(michaelrj): Fix the floating point dependencies
    OLD_FPUTIL_DEPS = [
        ":__support_fputil_basic_operations",
        ":__support_fputil_division_and_remainder_operations",
        ":__support_fputil_fenv_impl",
        ":__support_fputil_fp_bits",
        ":__support_fputil_float_properties",
        ":__support_fputil_hypot",
        ":__support_fputil_manipulation_functions",
        ":__support_fputil_nearest_integer_operations",
        ":__support_fputil_normal_float",
        ":__support_fputil_platform_defs",
        ":__support_builtin_wrappers",
        ":__support_fputil_except_value_utils",
    ]
    libc_function(
        name = name,
        srcs = selects.with_or(select_map),
        hdrs = ["src/math/" + name + ".h"],
        deps = [":__support_common"] + OLD_FPUTIL_DEPS + additional_deps,
    )
