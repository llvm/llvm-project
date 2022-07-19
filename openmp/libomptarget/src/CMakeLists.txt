##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
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
  ${CMAKE_CURRENT_SOURCE_DIR}/hostrpc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ompt_buffer_mgr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ompt_callback.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rtl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interop.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/omptarget.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rtl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LegacyAPI.cpp
)

set(LIBOMPTARGET_SRC_FILES ${LIBOMPTARGET_SRC_FILES} PARENT_SCOPE)

# Build libomptarget library with libdl dependency.
add_library(omptarget SHARED ${LIBOMPTARGET_SRC_FILES})
# Don't override an externally defined RPATH
if(NOT DEFINED CMAKE_INSTALL_RPATH)
  set_target_properties(omptarget PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib" BUILD_RPATH "$ORIGIN")
else()
  set_target_properties(omptarget PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_RPATH} BUILD_RPATH ${CMAKE_INSTALL_RPATH})
endif()
if (OPENMP_ENABLE_LIBOMPTARGET_PROFILING)
  # Add LLVMSupport dependency if profiling is enabled.
  # Linking with LLVM component libraries also requires
  # aligning the compile flags.
  llvm_update_compile_flags(omptarget)
  target_compile_definitions(omptarget PUBLIC OMPTARGET_PROFILE_ENABLED)
  target_link_libraries(omptarget PRIVATE LLVMSupport)
endif()

find_library(LLVM_OFFLOAD_ARCH LLVMOffloadArch HINTS ${LLVM_LIBRARY_DIR} REQUIRED)

target_include_directories(omptarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR})

# If LLVM DYLIB is off we need to link in libclang-cpp.
find_library(LLVM_DYLIB LLVM HINTS ${LLVM_LIBRARY_DIR})
if(NOT LLVM_DYLIB)
  libomptarget_say("LLVM_DYLIB off: linking libclang-cpp with libomptarget.so")
  find_library(CLANG_CPP clang-cpp HINTS ${LLVM_LIBRARY_DIR} REQUIRED)
  target_link_libraries(omptarget PRIVATE ${CLANG_CPP})
endif()

target_link_libraries(omptarget PRIVATE
  ${CMAKE_DL_LIBS}
  ${LLVM_OFFLOAD_ARCH}
  "-Wl,--no-allow-shlib-undefined"
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")

# Install libomptarget under the lib destination folder.
install(TARGETS omptarget LIBRARY COMPONENT omptarget
  DESTINATION "${RUN_PACKAGE}${OPENMP_INSTALL_LIBDIR}")
