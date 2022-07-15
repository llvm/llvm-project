##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build offloading library libomptarget.so.
#
##===----------------------------------------------------------------------===##

libomptarget_say("Building offloading runtime library libomptarget.")

add_llvm_library(omptarget
  SHARED

  api.cpp
  device.cpp
  interface.cpp
  interop.cpp
  omptarget.cpp
  rtl.cpp
  LegacyAPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}

  LINK_COMPONENTS
  Support
  Object

  LINK_LIBS
  ${CMAKE_DL_LIBS}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports"
)

target_include_directories(omptarget PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})

# Also install plugin under the standard lib destination folder.
install(TARGETS omptarget LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
