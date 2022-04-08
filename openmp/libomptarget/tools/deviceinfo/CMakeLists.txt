##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build llvm-omp-device-info tool
#
##===----------------------------------------------------------------------===##

libomptarget_say("Building the llvm-omp-device-info tool")

add_llvm_tool(llvm-omp-device-info llvm-omp-device-info.cpp)

llvm_update_compile_flags(llvm-omp-device-info)

target_link_libraries(llvm-omp-device-info PRIVATE
  omp
  omptarget
)
