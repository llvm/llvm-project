# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A macro to produce a loadable plugin library for the target OS.

This macro produces a set of platform-specific `cc_binary` rules, by appending
the platform suffix (`.dll`, `.dylib`, or `.so`) to the provided `name`. It then
connects these to a `cc_import` rule with `name` exactly and `hdrs` that can be
used by other Bazel rules to depend on the plugin library.

The `srcs` attribute for the `cc_binary` rules is `srcs + hdrs`. Other explicit
arguments are passed to all of the rules where they apply, and can be used to
configure generic aspects of all generated rules such as `testonly`. Lastly,
`kwargs` is expanded into all the `cc_binary` rules.
"""

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_import", "cc_library")

def cc_plugin_library(name, srcs, hdrs, include_prefix = None, strip_include_prefix = None, alwayslink = False, features = [], tags = [], testonly = False, **kwargs):
    # Neither the name of the plugin binary nor tags on whether it is built are
    # configurable. Instead, we build a `cc_binary` with each name and
    # selectively depend on them based on platform.
    #
    # All-in-all, this is a pretty poor workaround. I think this is part of the
    # Bazel issue: https://github.com/bazelbuild/bazel/issues/7538
    so_name = name + ".so"
    dll_name = name + ".dll"
    dylib_name = name + ".dylib"
    interface_output_name = name + "_interface_output"
    import_name = name + "_import"
    for impl_name in [dll_name, dylib_name, so_name]:
        cc_binary(
            name = impl_name,
            srcs = srcs + hdrs,
            linkshared = True,
            linkstatic = True,
            features = features,
            tags = ["manual"] + tags,
            testonly = testonly,
            **kwargs
        )
    native.filegroup(
        name = interface_output_name,
        srcs = select({
            "@platforms//os:windows": [":" + dll_name],
            "@platforms//os:macos": [":" + dylib_name],
            "//conditions:default": [":" + so_name],
        }),
        output_group = "interface_library",
    )
    cc_import(
        name = import_name,
        interface_library = ":" + interface_output_name,
        shared_library = select({
            "@platforms//os:windows": ":" + dll_name,
            "@platforms//os:macos": ":" + dylib_name,
            "//conditions:default": ":" + so_name,
        }),
        alwayslink = alwayslink,
        features = features,
        tags = tags,
        testonly = testonly,
    )
    cc_library(
        name = name,
        hdrs = hdrs,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        deps = [":" + import_name],
        alwayslink = alwayslink,
        features = features,
        tags = tags,
        testonly = testonly,
    )
