##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for an aarch64 machine if available.
#
##===----------------------------------------------------------------------===##

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  build_generic_elf64("aarch64" "aarch64" "aarch64" "aarch64-unknown-linux-gnu" "183")
else()
 libomptarget_say("Not building aarch64 offloading plugin: machine not found in the system.")
endif()
