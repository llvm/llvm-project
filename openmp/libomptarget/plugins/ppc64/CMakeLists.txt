##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
# 
##===----------------------------------------------------------------------===##
#
# Build a plugin for a ppc64 machine if available.
#
##===----------------------------------------------------------------------===##

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  build_generic_elf64("ppc64" "PPC64" "ppc64" "powerpc64-ibm-linux-gnu" "21")
else()
 libomptarget_say("Not building ppc64 offloading plugin: machine not found in the system.")
endif()