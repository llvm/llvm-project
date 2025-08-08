# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

package(
    default_visibility = ["//visibility:public"],
    # BSD/MIT-like license (for zlib)
    licenses = ["notice"],
)

bool_flag(
    name = "llvm_enable_zlib",
    build_setting_default = True,
)

config_setting(
    name = "llvm_zlib_enabled",
    flag_values = {":llvm_enable_zlib": "true"},
)

copy_file(
    # The input template is identical to the CMake output.
    name = "zconf_gen",
    src = "zconf.h.in",
    out = "zconf.h",
    allow_symlink = True,
)

cc_library(
    name = "zlib",
    srcs = select({
        ":llvm_zlib_enabled": [
            "adler32.c",
            "adler32_p.h",
            "chunkset.c",
            "chunkset_tpl.h",
            "compare258.c",
            "compress.c",
            "crc32.c",
            "crc32_comb.c",
            "crc32_comb_tbl.h",
            "crc32_p.h",
            "crc32_tbl.h",
            "deflate.c",
            "deflate.h",
            "deflate_fast.c",
            "deflate_medium.c",
            "deflate_p.h",
            "deflate_quick.c",
            "deflate_slow.c",
            "fallback_builtins.h",
            "functable.c",
            "functable.h",
            "infback.c",
            "inffast.c",
            "inffast.h",
            "inffixed_tbl.h",
            "inflate.c",
            "inflate.h",
            "inflate_p.h",
            "inftrees.c",
            "inftrees.h",
            "insert_string.c",
            "insert_string_tpl.h",
            "match_tpl.h",
            "trees.c",
            "trees.h",
            "trees_emit.h",
            "trees_tbl.h",
            "uncompr.c",
            "zbuild.h",
            "zendian.h",
            "zutil.c",
            "zutil.h",
            "zutil_p.h",
        ],
        "//conditions:default": [],
    }),
    hdrs = select({
        ":llvm_zlib_enabled": [
            "zlib.h",
            ":zconf_gen",
        ],
        "//conditions:default": [],
    }),
    copts = [
        "-std=c11",
        "-DZLIB_COMPAT",
        "-DWITH_GZFILEOP",
        "-DWITH_OPTIM",
        "-DWITH_NEW_STRATEGIES",
        # For local builds you might want to add "-DWITH_NATIVE_INSTRUCTIONS"
        # here to improve performance. Native instructions aren't enabled in
        # the default config for reproducibility.
    ],
    defines = select({
        ":llvm_zlib_enabled": [
            "LLVM_ENABLE_ZLIB=1",
        ],
        "//conditions:default": [],
    }),
    # Clang includes zlib with angled instead of quoted includes, so we need
    # strip_include_prefix here.
    strip_include_prefix = ".",
    visibility = ["//visibility:public"],
)
