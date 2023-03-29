# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for zstd)
    licenses = ["notice"],
)

bool_flag(
    name = "llvm_enable_zstd",
    build_setting_default = True,
)

config_setting(
    name = "llvm_zstd_enabled",
    flag_values = {":llvm_enable_zstd": "true"},
)

cc_library(
    name = "zstd",
    srcs = select({
        ":llvm_zstd_enabled": glob([
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
        ":llvm_zstd_enabled": [
            "lib/zstd.h",
            "lib/zdict.h",
            "lib/zstd_errors.h",
        ],
        "//conditions:default": [],
    }),
    defines = select({
        ":llvm_zstd_enabled": [
            "LLVM_ENABLE_ZSTD=1",
            "ZSTD_MULTITHREAD",
        ],
        "//conditions:default": [],
    }),
    strip_include_prefix = "lib",
)
