# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc starlark rules for building individual functions."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_skylib//lib:selects.bzl", "selects")
load(":libc_configure_options.bzl", "LIBC_CONFIGURE_OPTIONS")
load(":libc_namespace.bzl", "LIBC_NAMESPACE")
load(":platforms.bzl", "PLATFORM_CPU_X86_64")

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

def _libc_library(name, **kwargs):
    """Internal macro to serve as a base for all other libc library rules.

    Args:
      name: Target name.
      **kwargs: All other attributes relevant for the cc_library rule.
    """

    for attr in ["copts", "local_defines"]:
        if attr in kwargs:
            fail("disallowed attribute: '{}' in rule: '{}'".format(attr, name))
    native.cc_library(
        name = name,
        copts = libc_common_copts(),
        local_defines = LIBC_CONFIGURE_OPTIONS,
        linkstatic = 1,
        **kwargs
    )

# A convenience function which should be used to list all libc support libraries.
# Any library which does not define a public function should be listed with
# libc_support_library.
def libc_support_library(name, **kwargs):
    _libc_library(name = name, **kwargs)

def libc_function(name, **kwargs):
    """Add target for a libc function.

    This macro creates an internal cc_library that can be used to test this
    function.

    Args:
      name: Target name. Typically the name of the function this target is for.
      **kwargs: Other attributes relevant for a cc_library. For example, deps.
    """

    # Builds "internal" library with a function, exposed as a C++ function in
    # the "LIBC_NAMESPACE" namespace. This allows us to test the function in the
    # presence of another libc.
    _libc_library(name = name, **kwargs)

LibcLibraryInfo = provider(
    "All source files and textual headers for building a particular library.",
    fields = ["srcs", "textual_hdrs"],
)

def _get_libc_info_aspect_impl(
        target,  # @unused
        ctx):
    maybe_srcs = getattr(ctx.rule.attr, "srcs", [])
    maybe_hdrs = getattr(ctx.rule.attr, "hdrs", [])
    maybe_textual_hdrs = getattr(ctx.rule.attr, "textual_hdrs", [])
    maybe_deps = getattr(ctx.rule.attr, "deps", [])
    return LibcLibraryInfo(
        srcs = depset(
            transitive = [
                dep[LibcLibraryInfo].srcs
                for dep in maybe_deps
                if LibcLibraryInfo in dep
            ] + [
                src.files
                for src in maybe_srcs + maybe_hdrs
            ],
        ),
        textual_hdrs = depset(
            transitive = [
                dep[LibcLibraryInfo].textual_hdrs
                for dep in maybe_deps
                if LibcLibraryInfo in dep
            ] + [
                hdr.files
                for hdr in maybe_textual_hdrs
            ],
        ),
    )

_get_libc_info_aspect = aspect(
    implementation = _get_libc_info_aspect_impl,
    attr_aspects = ["deps"],
)

def _libc_srcs_filegroup_impl(ctx):
    srcs = depset(transitive = [
        fn[LibcLibraryInfo].srcs
        for fn in ctx.attr.libs
    ])
    if ctx.attr.enforce_headers_only:
        paths = [f.short_path for f in srcs.to_list() if f.extension != "h"]
        if paths:
            fail("Unexpected non-header files: {}".format(paths))
    return DefaultInfo(files = srcs)

_libc_srcs_filegroup = rule(
    doc = "Returns all sources for building the specified libraries.",
    implementation = _libc_srcs_filegroup_impl,
    attrs = {
        "libs": attr.label_list(
            mandatory = True,
            aspects = [_get_libc_info_aspect],
        ),
        "enforce_headers_only": attr.bool(default = False),
    },
)

def _libc_textual_hdrs_filegroup_impl(ctx):
    return DefaultInfo(
        files = depset(transitive = [
            fn[LibcLibraryInfo].textual_hdrs
            for fn in ctx.attr.libs
        ]),
    )

_libc_textual_hdrs_filegroup = rule(
    doc = "Returns all textual headers for compiling the specified libraries.",
    implementation = _libc_textual_hdrs_filegroup_impl,
    attrs = {
        "libs": attr.label_list(
            mandatory = True,
            aspects = [_get_libc_info_aspect],
        ),
    },
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

    _libc_srcs_filegroup(
        name = name + "_srcs",
        libs = libc_functions,
    )

    _libc_textual_hdrs_filegroup(
        name = name + "_textual_hdrs",
        libs = libc_functions,
    )
    native.cc_library(
        name = name + "_textual_hdr_library",
        textual_hdrs = [":" + name + "_textual_hdrs"],
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

def libc_header_library(name, hdrs, deps = [], **kwargs):
    """Creates a header-only library to share libc functionality.

    Args:
      name: Name of the cc_library target.
      hdrs: List of headers to be shared.
      deps: The list of libc_support_library dependencies if any.
      **kwargs: All other attributes relevant for the cc_library rule.
    """

    _libc_srcs_filegroup(
        name = name + "_hdr_deps",
        libs = deps,
        enforce_headers_only = True,
    )

    _libc_textual_hdrs_filegroup(
        name = name + "_textual_hdrs",
        libs = deps,
    )
    native.cc_library(
        name = name + "_textual_hdr_library",
        textual_hdrs = [":" + name + "_textual_hdrs"],
    )

    native.cc_library(
        name = name,
        hdrs = hdrs,
        # We put _hdr_deps in srcs, as they are not a part of this cc_library
        # interface, but instead are used to implement shared headers.
        srcs = [":" + name + "_hdr_deps"],
        deps = [":" + name + "_textual_hdr_library"],
        # copts don't really matter, since it's a header-only library, but we
        # need proper -I flags for header validation, which are specified in
        # libc_common_copts().
        copts = libc_common_copts(),
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
