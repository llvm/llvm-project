##===----------------------------------------------------------------------===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# Build a plugin for a x86_64 machine if available.
#
##===----------------------------------------------------------------------===##

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  build_generic_elf64("x86_64" "x86_64" "x86_64" "x86_64-pc-linux-gnu" "62")
else()
 libomptarget_say("Not building x86_64 offloading plugin: machine not found in the system.")
endif()
