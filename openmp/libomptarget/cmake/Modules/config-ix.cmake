#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# Checking C, CXX
check_cxx_compiler_flag(-std=c++11 LIBOMPTARGET_HAVE_STD_CPP11_FLAG)
check_c_compiler_flag(-Werror LIBOMPTARGET_HAVE_WERROR_FLAG)
