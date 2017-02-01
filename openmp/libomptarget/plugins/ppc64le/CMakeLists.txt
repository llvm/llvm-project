##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
# 
##===----------------------------------------------------------------------===##
#
# Build a plugin for a ppc64le machine if available.
#
##===----------------------------------------------------------------------===##

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  build_generic_elf64("ppc64le" "PPC64le" "ppc64" "powerpc64le-ibm-linux-gnu" "21")
else()
 libomptarget_say("Not building ppc64le offloading plugin: machine not found in the system.")
endif()