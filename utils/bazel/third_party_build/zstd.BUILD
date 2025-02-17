# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for zstd)
    licenses = ["notice"],
)

cc_library(
    name = "zstd",
    srcs = glob([
        "lib/common/*.c",
        "lib/common/*.h",
        "lib/compress/*.c",
        "lib/compress/*.h",
        "lib/decompress/*.c",
        "lib/decompress/*.h",
        "lib/decompress/*.S",
        "lib/dictBuilder/*.c",
        "lib/dictBuilder/*.h",
    ]),
    hdrs = [
        "lib/zdict.h",
        "lib/zstd.h",
        "lib/zstd_errors.h",
    ],
    defines = [
        "ZSTD_MULTITHREAD",
    ],
    strip_include_prefix = "lib",
)
