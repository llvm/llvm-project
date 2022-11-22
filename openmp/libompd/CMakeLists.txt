#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#// Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#// Notified per clause 4(b) of the license.
#//
#//===----------------------------------------------------------------------===//
#

find_package (Python3 COMPONENTS Interpreter Development)
option(LIBOMP_OMPD_GDB_SUPPORT "Build gdb-plugin code that enables OMPD support in GDB." ${Python3_FOUND})

if(LIBOMP_OMPD_SUPPORT)
    set(OMPD_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src/)
    add_subdirectory(src)

    # HWLOC-support
    set(LIBOMP_USE_HWLOC FALSE CACHE BOOL
      "Use Hwloc (http://www.open-mpi.org/projects/hwloc/) library for affinity?")
    set(LIBOMP_HWLOC_INSTALL_DIR /usr/local CACHE PATH
      "Install path for hwloc library")

    if(LIBOMP_OMPD_GDB_SUPPORT)
        add_subdirectory(gdb-plugin)
        # temporarily disabled
        # test/CMakeLists.txt breaks DeviceRTL in LLVM_ENABLE_PROJECTS builds.
        # find_package(LLVM) is likely the source of the trouble.
        # add_subdirectory(test)
    endif()
endif()
