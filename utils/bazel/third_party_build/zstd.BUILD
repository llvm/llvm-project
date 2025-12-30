# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for zstd)
    licenses = ["notice"],
)

cc_library(
    name = "zstd",
    srcs = select({
        "@llvm-project//third-party:llvm_zstd_enabled": glob([
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
        "//conditions:default": [],
    }),
    hdrs = select({
        "@llvm-project//third-party:llvm_zstd_enabled": [
            "lib/zdict.h",
            "lib/zstd.h",
            "lib/zstd_errors.h",
        ],
        "//conditions:default": [],
    }),
    defines = select({
        "@llvm-project//third-party:llvm_zstd_enabled": [
            "LLVM_ENABLE_ZSTD=1",
            "ZSTD_MULTITHREAD",
        ],
        "//conditions:default": [],
    }),
    strip_include_prefix = "lib",
)
