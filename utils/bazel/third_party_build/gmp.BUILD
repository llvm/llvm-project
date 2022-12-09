# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make_variant")

filegroup(
    name = "sources",
    srcs = glob(["**"]),
)

configure_make_variant(
    name = "gmp",
    configure_options = ["--with-pic"],
    copts = ["-w"],
    lib_name = "libgmp",
    lib_source = ":sources",
    toolchain = "@rules_foreign_cc//toolchains:preinstalled_autoconf_toolchain",
    visibility = ["//visibility:public"],
)
