##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a s390x machine if available.
#
##===----------------------------------------------------------------------===##

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
 build_generic_elf64("SystemZ" "S390X" "s390x" "s390x-ibm-linux-gnu" "22")
else()
 libomptarget_say("Not building s390x NextGen offloading plugin: machine not found in the system.")
endif()
