# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_foreign_cc//foreign_cc:defs.bzl", "make_variant")

filegroup(
    name = "sources",
    srcs = glob(["**"]),
)

make_variant(
    name = "pfm",
    copts = ["-w"],
    lib_name = "libpfm",
    lib_source = ":sources",
    target_compatible_with = select({
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    toolchain = "@rules_foreign_cc//toolchains:preinstalled_autoconf_toolchain",
    visibility = ["//visibility:private"],
)

alias(
    name = "pfm_external",
    actual = "@pfm//:pfm_",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pfm_system",
    linkopts = ["-lpfm"],
    target_compatible_with = select({
        "@platforms//os:linux": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)
