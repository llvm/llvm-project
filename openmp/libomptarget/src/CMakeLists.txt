##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
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
  rtl.cpp
  omptarget.cpp
)

# Build libomptarget library with libdl dependency.
add_library(omptarget SHARED ${src_files})
target_link_libraries(omptarget
  ${CMAKE_DL_LIBS}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")

# Install libomptarget under the lib destination folder.
install(TARGETS omptarget LIBRARY DESTINATION lib${OPENMP_LIBDIR_SUFFIX})
