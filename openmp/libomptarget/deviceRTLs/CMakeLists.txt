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
# Currently device libm functions are created by clang headers.  These do not
# work for FORTRAN. The libm here  builds a libm device library by turning on
# __BUILD_MATH_BUILTINS_LIB__ This converts the math functions in the clang
# headers from static to extern to build device linkable libm for FORTRAN.
# Eventually, we can make a linkable device libm for c and c++ and remove a
# lot of header definitions that get compiled with every offload compilation.
add_subdirectory(libm)
