##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
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