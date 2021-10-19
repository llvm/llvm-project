# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper library for the system's zlib. Using this only works if the toolchain
# already has the relevant header search and library search paths configured.
# It also sets the relevant LLVM `#define`s to enable zlib.
cc_library(
    name = "zlib",
    defines = ["LLVM_ENABLE_ZLIB=1"],
    linkopts = ["-lz"],
    visibility = ["//visibility:public"],
)
