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

set(LIBOMPTARGET_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interop.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/omptarget.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rtl.cpp
)

set(LIBOMPTARGET_SRC_FILES ${LIBOMPTARGET_SRC_FILES} PARENT_SCOPE)

include_directories(${LIBOMPTARGET_LLVM_INCLUDE_DIRS})

# Build libomptarget library with libdl dependency.
add_library(omptarget SHARED ${LIBOMPTARGET_SRC_FILES})
set_target_properties(omptarget PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN")
if (OPENMP_ENABLE_LIBOMPTARGET_PROFILING)
  # Add LLVMSupport dependency if profiling is enabled.
  # Linking with LLVM component libraries also requires
  # aligning the compile flags.
  llvm_update_compile_flags(omptarget)
  target_compile_definitions(omptarget PUBLIC OMPTARGET_PROFILE_ENABLED)
  target_link_libraries(omptarget PRIVATE LLVMSupport)
endif()
target_link_libraries(omptarget PRIVATE
  ${CMAKE_DL_LIBS}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")

# Install libomptarget under the lib destination folder.
install(TARGETS omptarget LIBRARY COMPONENT omptarget
  DESTINATION "${OPENMP_INSTALL_LIBDIR}")
