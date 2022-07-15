##===----------------------------------------------------------------------===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
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

# Don't override an externally defined RPATH
if(NOT DEFINED CMAKE_INSTALL_RPATH)
  set_property(TARGET omptarget.rtl.x86_64 PROPERTY INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib")
else()
  set_target_properties(omptarget.rtl.x86_64 PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_RPATH}
    BUILD_RPATH ${CMAKE_INSTALL_RPATH})
endif()
