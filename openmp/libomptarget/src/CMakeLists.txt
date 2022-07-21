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

add_llvm_library(omptarget
  SHARED

  api.cpp
  device.cpp
  interface.cpp
  hostrpc.cpp
  ompt_buffer_mgr.cpp
  ompt_callback.cpp
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
  #  NO_INSTALL_RPATH
  )

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

target_include_directories(omptarget PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})

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

# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
install(TARGETS omptarget LIBRARY COMPONENT omptarget DESTINATION "${RUN_PACKAGE}${OPENMP_INSTALL_LIBDIR}")
