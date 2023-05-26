# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for details.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===----------------------------------------------------------------------===//
  
set(LIBOMP_ARCHER_SUPPORT TRUE CACHE BOOL "Build libomp with archer support")

if(LIBOMP_OMPT_SUPPORT AND LIBOMP_ARCHER_SUPPORT)
  include_directories(${LIBOMP_INCLUDE_DIR})

  add_library(archer SHARED ompt-tsan.cpp)
  target_link_libraries(archer ${CMAKE_DL_LIBS})
  add_library(archer_static STATIC ompt-tsan.cpp)

  install(TARGETS archer archer_static
    LIBRARY DESTINATION ${OPENMP_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${OPENMP_INSTALL_LIBDIR})

  add_subdirectory(tests)
endif()
