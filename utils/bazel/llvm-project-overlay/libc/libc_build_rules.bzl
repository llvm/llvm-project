# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc starlark rules for building individual functions."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_skylib//lib:selects.bzl", "selects")
load(":libc_configure_options.bzl", "LIBC_CONFIGURE_OPTIONS")
load(":libc_namespace.bzl", "LIBC_NAMESPACE")
load(":platforms.bzl", "PLATFORM_CPU_X86_64")

def libc_internal_target(name):
    return name + ".__internal__"

def libc_common_copts():
    root_label = Label(":libc")
    libc_include_path = paths.join(root_label.workspace_root, root_label.package)
    return [
        "-I" + libc_include_path,
        "-I" + paths.join(libc_include_path, "include"),
        "-DLIBC_NAMESPACE=" + LIBC_NAMESPACE,
    ]

def libc_release_copts():
    copts = [
        "-DLIBC_COPT_PUBLIC_PACKAGING",
        # This is used to explicitly give public symbols "default" visibility.
        # See src/__support/common.h for more information.
        "-DLLVM_LIBC_FUNCTION_ATTR='[[gnu::visibility(\"default\")]]'",
        # All other libc sources need to be compiled with "hidden" visibility.
        "-fvisibility=hidden",
        "-O3",
        "-fno-builtin",
        "-fno-lax-vector-conversions",
        "-ftrivial-auto-var-init=pattern",
        "-fno-omit-frame-pointer",
        "-fstack-protector-strong",
    ]

    platform_copts = selects.with_or({
        PLATFORM_CPU_X86_64: ["-mno-omit-leaf-frame-pointer"],
        "//conditions:default": [],
    })
    return copts + platform_copts

def _libc_library(name, copts = [], deps = [], local_defines = [], **kwargs):
    """Internal macro to serve as a base for all other libc library rules.

    Args:
      name: Target name.
      copts: The special compiler options for the target.
      deps: The list of target dependencies if any.
      local_defines: The list of target local_defines if any.
      **kwargs: All other attributes relevant for the cc_library rule.
    """

    native.cc_library(
        name = name,
        copts = copts + libc_common_copts(),
        local_defines = local_defines + LIBC_CONFIGURE_OPTIONS,
        deps = deps,
        linkstatic = 1,
        **kwargs
    )

def _libc_library_filegroups(
        name,
        is_function,
        srcs = [],
        hdrs = [],
        textual_hdrs = [],
        deps = [],
        # We're not using kwargs, but instead explicitly list all possible
        # arguments that can be passed to libc_support_library or
        # libc_function macros. This is done to limit the configurability
        # and ensure the consistent and tightly controlled set of flags
        # (see libc_common_copts and libc_release_copts above) is used to build
        # libc code both for tests and for release configuration.
        target_compatible_with = None,  # @unused
        weak = False):  # @unused
    """Internal macro to collect sources and headers required to build a library.
    """

    # filegroups created from "libc_function" macro has an extra "_fn" in their
    # name to ensure that no other libc target can depend on libc_function.
    prefix = name + ("_fn" if is_function else "")
    native.filegroup(
        name = prefix + "_srcs",
        srcs = srcs + hdrs + [dep + "_srcs" for dep in deps],
    )
    native.filegroup(
        name = prefix + "_textual_hdrs",
        srcs = textual_hdrs + [dep + "_textual_hdrs" for dep in deps],
    )

# A convenience function which should be used to list all libc support libraries.
# Any library which does not define a public function should be listed with
# libc_support_library.
def libc_support_library(name, **kwargs):
    _libc_library(name = name, **kwargs)
    _libc_library_filegroups(name = name, is_function = False, **kwargs)

def libc_function(
        name,
        weak = False,
        **kwargs):
    """Add target for a libc function.

    This macro creates an internal cc_library that can be used to test this
    function, and creates filegroups required to include this function into
    a release build of libc.

    Args:
      name: Target name. It is normally the name of the function this target is
            for.
      weak: Make the symbol corresponding to the libc function "weak".
      **kwargs: Other attributes relevant for a cc_library. For example, deps.
    """

    # Build "internal" library with a function, the target has ".__internal__" suffix and contains
    # C++ functions in the "LIBC_NAMESPACE" namespace. This allows us to test the function in the
    # presence of another libc.
    _libc_library(
        name = libc_internal_target(name),
        **kwargs
    )

    _libc_library_filegroups(name = name, is_function = True, **kwargs)


    # TODO(PR #130327): Remove this after downstream uses are migrated to libc_release_library.
    # This second target is the llvm libc C function with default visibility.
    func_attrs = [
        "LLVM_LIBC_FUNCTION_ATTR_" + name + "='LLVM_LIBC_EMPTY, [[gnu::weak]]'",
    ] if weak else []

    _libc_library(
        name = name,
        copts = libc_release_copts(),
        local_defines = func_attrs,
        **kwargs
    )

def libc_release_library(
        name,
        libc_functions,
        weak_symbols = [],
        **kwargs):
    """Create the release version of a libc library.

    Args:
        name: Name of the cc_library target.
        libc_functions: List of functions to include in the library. They should be
            created by libc_function macro.
        weak_symbols: List of function names that should be marked as weak symbols.
        **kwargs: Other arguments relevant to cc_library.
    """
    # Combine all sources into a single filegroup to avoid repeated sources error.
    native.filegroup(
        name = name + "_srcs",
        srcs = [function + "_fn_srcs" for function in libc_functions],
    )

    native.cc_library(
        name = name + "_textual_hdr_library",
        textual_hdrs = [function + "_fn_textual_hdrs" for function in libc_functions],
    )

    weak_attributes = [
        "LLVM_LIBC_FUNCTION_ATTR_" + name + "='LLVM_LIBC_EMPTY, [[gnu::weak]]'"
        for name in weak_symbols
    ]

    native.cc_library(
        name = name,
        srcs = [":" + name + "_srcs"],
        copts = libc_common_copts() + libc_release_copts(),
        local_defines = weak_attributes + LIBC_CONFIGURE_OPTIONS,
        deps = [
            ":" + name + "_textual_hdr_library",
        ],
        **kwargs
    )

def libc_math_function(
        name,
        additional_deps = None):
    """Add a target for a math function.

    Args:
      name: The name of the function.
      additional_deps: Other deps like helper cc_library targets used by the
                       math function.
    """
    additional_deps = additional_deps or []

    #TODO(michaelrj): Fix the floating point dependencies
    OLD_FPUTIL_DEPS = [
        ":__support_fputil_basic_operations",
        ":__support_fputil_division_and_remainder_operations",
        ":__support_fputil_fenv_impl",
        ":__support_fputil_fp_bits",
        ":__support_fputil_hypot",
        ":__support_fputil_manipulation_functions",
        ":__support_fputil_nearest_integer_operations",
        ":__support_fputil_normal_float",
        ":__support_math_extras",
        ":__support_fputil_except_value_utils",
    ]
    libc_function(
        name = name,
        srcs = ["src/math/generic/" + name + ".cpp"],
        hdrs = ["src/math/" + name + ".h"],
        deps = [":__support_common"] + OLD_FPUTIL_DEPS + additional_deps,
    )
