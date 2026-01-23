# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for PyYAML)
    licenses = ["notice"],
)

py_library(
    name = "yaml",
    srcs = glob(["yaml/*.py"]),
)
