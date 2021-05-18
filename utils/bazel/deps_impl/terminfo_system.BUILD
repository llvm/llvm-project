# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Wrapper library for some system terminfo. Using this only works if the
# toolchain already has the relevant library search paths configured. It also
# sets the relevant LLVM `#define`s to enoble using terminfo.
cc_library(
    name = "terminfo",
    defines = ["LLVM_ENABLE_TERMINFO=1"],
    # Note that we will replace these link options with ones needed to
    # effectively link against a terminfo providing library on the system.
    linkopts = {TERMINFO_LINKOPTS},
    visibility = ["//visibility:public"],
)
