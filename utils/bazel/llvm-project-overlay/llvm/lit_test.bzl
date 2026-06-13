# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Rules for running lit tests."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_python//python:defs.bzl", _py_test = "py_test")

def lit_test(
        name,
        srcs,
        args = None,
        data = None,
        deps = None,
        py_test = _py_test,
        **kwargs):
    """Runs a single test file with LLVM's lit tool.

    Args:
      name: string. the name of the generated test target.
      srcs: label list. The files on which to run lit.
      args: string list. Additional arguments to pass to lit.
        Note that `-v` and the 'srcs' paths are added automatically.
      data: label list. Additional data dependencies of the test.
        Note that 'srcs' targets are added automatically.
      deps: label list. List of targets the test depends on.
      py_test: function. The py_test rule to use for the underlying test.
      **kwargs: additional keyword arguments.

    See https://llvm.org/docs/CommandGuide/lit.html for details on lit.
    """

    args = args or []
    data = data or []
    deps = deps or []
    py_test(
        name = name,
        srcs = [Label("//llvm:lit")],
        main = Label("//llvm:utils/lit/lit.py"),
        args = args + ["-v"] + ["$(rootpath %s)" % src for src in srcs],
        data = data + srcs,
        legacy_create_init = False,
        deps = deps + [Label("//llvm:lit")],
        **kwargs
    )


def runfiles_path(label):
    """Returns the path relative to execution CWD of a runnable target (run/test)

    Args:
      label: Label. The label to return the runfiles path of.

    For example, runfiles_path("@foo//bar:BUILD") returns `../foo/bar`.
    """

    # https://bazel.build/remote/output-directories#layout-diagram
    # When running tests, the current working directory is  <testxyz.runfiles>/_main
    # The runfiles for external modules are located in <testxyz.runfiles>/<external_mod.workspace_name>/<pkg_path>
    rfiles_path = "../" + paths.join(Label(label).workspace_name, Label(label).package)
    rfiles_path = paths.normalize(rfiles_path)
    return rfiles_path
