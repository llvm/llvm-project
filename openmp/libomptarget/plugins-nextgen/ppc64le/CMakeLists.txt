##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a ppc64le machine if available.
#
##===----------------------------------------------------------------------===##

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  build_generic_elf64_nextgen("ppc64le" "PPC64le" "ppc64" "powerpc64le-ibm-linux-gnu" "21")
else()
 libomptarget_say("Not building ppc64le NextGen offloading plugin: machine not found in the system.")
endif()
