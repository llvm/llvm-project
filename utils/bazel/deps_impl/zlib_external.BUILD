# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper around an external zlib library to add the relevant LLVM `#define`s.
cc_library(
    name = "zlib",
    defines = ["LLVM_ENABLE_ZLIB=1"],
    visibility = ["//visibility:public"],
    deps = ["@external_zlib_repo//:zlib_rule"],
)
