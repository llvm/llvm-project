# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A macro to produce a loadable plugin library for the target OS.

This macro produces a `cc_binary` rule with the name `name + "_impl"`. It
forces the rule to statically link in its dependencies but to be linked as a
shared "plugin" library. It then creates binary aliases to `.so`, `.dylib`
,and `.dll` suffixed names for use on various platforms and selects between
these into a filegroup with the exact name passed to the macro.
"""

load("@rules_cc//cc:defs.bzl", "cc_binary")
load(":binary_alias.bzl", "binary_alias")

def cc_plugin_library(name, **kwargs):
    # Neither the name of the plugin binary nor tags on whether it is built are
    # configurable. Instead, we build a `cc_binary` that implements the plugin
    # library using a `_impl` suffix. Bazel will use appropriate flags to cause
    # this file to be a plugin library regardless of its name. We then create
    # binary aliases in the different possible platform names, and select
    # between these different names into a filegroup. The macro's name becomes
    # the filegroup name and it contains exactly one target that is the target
    # platform suffixed plugin library.
    #
    # All-in-all, this is a pretty poor workaround. I think this is part of the
    # Bazel issue: https://github.com/bazelbuild/bazel/issues/7538
    cc_binary(
        name = name + "_impl",
        linkshared = True,
        linkstatic = True,
        **kwargs
    )
    binary_alias(
        name = name + ".so",
        binary = ":" + name + "_impl",
    )
    binary_alias(
        name = name + ".dll",
        binary = ":" + name + "_impl",
    )
    binary_alias(
        name = name + ".dylib",
        binary = ":" + name + "_impl",
    )
    native.filegroup(
        name = name,
        srcs = select({
            "@bazel_tools//src/conditions:windows": [":" + name + ".dll"],
            "@bazel_tools//src/conditions:darwin": [":" + name + ".dylib"],
            "//conditions:default": [":" + name + ".so"],
        }),
    )
