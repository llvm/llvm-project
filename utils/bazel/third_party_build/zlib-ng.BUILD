# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")

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

genrule(
    # The input template is identical to the CMake output.
    name = "zconf_gen",
    srcs = ["zconf.h.in"],
    outs = ["zconf.h"],
    cmd = "cp $(SRCS) $(OUTS)",
)

cc_library(
    name = "zlib",
    srcs = select({
        ":llvm_zlib_enabled": [
            "adler32_p.h",
            "chunkset_tpl.h",
            "crc32_p.h",
            "crc32_tbl.h",
            "crc32_comb_tbl.h",
            "deflate.h",
            "deflate_p.h",
            "functable.h",
            "fallback_builtins.h",
            "inffast.h",
            "inffixed_tbl.h",
            "inflate.h",
            "inflate_p.h",
            "inftrees.h",
            "insert_string_tpl.h",
            "match_tpl.h",
            "trees.h",
            "trees_emit.h",
            "trees_tbl.h",
            "zbuild.h",
            "zendian.h",
            "zutil.h",
            "adler32.c",
            "chunkset.c",
            "compare258.c",
            "compress.c",
            "crc32.c",
            "crc32_comb.c",
            "deflate.c",
            "deflate_fast.c",
            "deflate_medium.c",
            "deflate_quick.c",
            "deflate_slow.c",
            "functable.c",
            "infback.c",
            "inffast.c",
            "inflate.c",
            "inftrees.c",
            "insert_string.c",
            "trees.c",
            "uncompr.c",
            "zutil_p.h",
            "zutil.c",
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
