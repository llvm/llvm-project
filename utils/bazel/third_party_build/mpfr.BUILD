# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("@rules_foreign_cc//foreign_cc:defs.bzl", "configure_make_variant")

filegroup(
    name = "sources",
    srcs = glob(["**"]),
)

configure_make_variant(
    name = "mpfr",
    configure_options = ["--with-pic"],
    copts = ["-w"],
    lib_name = "libmpfr",
    lib_source = ":sources",
    toolchain = "@rules_foreign_cc//toolchains:preinstalled_autoconf_toolchain",
    visibility = ["//visibility:public"],
    deps = ["@gmp//:gmp_"],
)

alias(
    name = "mpfr_external",
    actual = "@mpfr//:mpfr_",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mpfr_system",
    linkopts = ["-lmpfr"],
    visibility = ["//visibility:public"],
)
