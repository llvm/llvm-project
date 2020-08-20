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

set(src_files
  api.cpp
  device.cpp
  interface.cpp
  MemoryManager.cpp
  rtl.cpp
  omptarget.cpp
)

# Build libomptarget library with libdl dependency.
add_library(omptarget SHARED ${src_files})
target_link_libraries(omptarget
  ${CMAKE_DL_LIBS}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")

# Install libomptarget under the lib destination folder.
install(TARGETS omptarget LIBRARY COMPONENT omptarget
  DESTINATION "${OPENMP_INSTALL_LIBDIR}")
