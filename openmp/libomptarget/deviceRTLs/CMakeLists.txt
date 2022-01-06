##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ##===----------------------------------------------------------------------===##
#
# Build a device RTL for each available machine.
#
##===----------------------------------------------------------------------===##

add_subdirectory(amdgcn)
add_subdirectory(nvptx)
