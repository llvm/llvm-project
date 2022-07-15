# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for details.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
# //
# //===----------------------------------------------------------------------===//
  
  

if(LIBOMP_OMPT_SUPPORT)
  include_directories(${LIBOMP_INCLUDE_DIR})

  add_library(archer SHARED ompt-tsan.cpp)
  add_library(archer_static STATIC ompt-tsan.cpp)

  install(TARGETS archer archer_static
    LIBRARY DESTINATION ${RUN_PACKAGE}${OPENMP_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${RUN_PACKAGE}${OPENMP_INSTALL_LIBDIR})

  add_subdirectory(tests)
endif()
