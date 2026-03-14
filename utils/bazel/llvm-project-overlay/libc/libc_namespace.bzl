# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLVM libc namespace declaration."""

load("//:vars.bzl", "LLVM_VERSION_MAJOR", "LLVM_VERSION_MINOR", "LLVM_VERSION_PATCH")

# The default libc namespace that encloses all functions.
LIBC_NAMESPACE = "__llvm_libc_{}_{}_{}_git".format(LLVM_VERSION_MAJOR, LLVM_VERSION_MINOR, LLVM_VERSION_PATCH)
