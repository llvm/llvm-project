#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

if(LIBOMP_OMPD_SUPPORT)
    set(OMPD_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/)
    add_subdirectory(src)
    if(NOT DISABLE_OMPD_GDB_PLUGIN)
        add_subdirectory(gdb-plugin)
    endif()
endif()
