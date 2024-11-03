# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc:defs.bzl", "cc_library")

_common_library_deps = [
    "//clang:ast",
    "//clang:ast_matchers",
    "//clang:basic",
    "//clang:lex",
    "//clang:frontend",
    "//llvm:FrontendOpenMP",
    "//llvm:Support",
]

def clang_tidy_library(name, **kwargs):
    kwargs["srcs"] = kwargs.get("srcs", native.glob([paths.join(name, "*.cpp")]))
    kwargs["hdrs"] = kwargs.get("hdrs", native.glob([paths.join(name, "*.h")]))
    kwargs["deps"] = kwargs.get("deps", []) + _common_library_deps
    cc_library(
        name = name,
        alwayslink = True,
        **kwargs
    )
