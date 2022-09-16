# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Rules for running lit tests."""

load("@bazel_skylib//lib:paths.bzl", "paths")

def lit_test(
        name,
        srcs,
        args = None,
        data = None,
        **kwargs):
    """Runs a single test file with LLVM's lit tool.

    Args:
      name: string. the name of the generated test target.
      srcs: label list. The files on which to run lit.
      args: string list. Additional arguments to pass to lit.
        Note that `-v` and the 'srcs' paths are added automatically.
      data: label list. Additional data dependencies of the test.
        Note that 'srcs' targets are added automatically.
      **kwargs: additional keyword arguments.

    See https://llvm.org/docs/CommandGuide/lit.html for details on lit.
    """

    args = args or []
    data = data or []

    native.py_test(
        name = name,
        srcs = ["//llvm:lit"],
        main = "//llvm:utils/lit/lit.py",
        args = args + ["-v"] + ["$(execpath %s)" % src for src in srcs],
        data = data + srcs,
        legacy_create_init = False,
        **kwargs
    )

def package_path(label):
    """Returns the path to the package of 'label'.

    Args:
      label: label. The label to return the package path of.

    For example, package_path("@foo//bar:BUILD") returns 'external/foo/bar'.
    """
    return paths.join(Label(label).workspace_root, Label(label).package)
