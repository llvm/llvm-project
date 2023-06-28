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
# libhsakmt.a
find_library ( HSAKMT_LIB libhsakmt.a REQURIED HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
get_filename_component ( HSAKMT_LIB_PATH ${HSAKMT_LIB} DIRECTORY )
link_directories (${HSAKMT_LIB_PATH})

# lib_amdgpu
pkg_check_modules(drm_amdgpu REQUIRED IMPORTED_TARGET libdrm_amdgpu)

# libnuma
find_library(numa libnuma.so REQUIRED HINTS /usr/lib/x86_64-linux-gnu/)

add_llvm_library(omptarget
  SHARED

  api.cpp
  device.cpp
  interface.cpp
  interop.cpp
  omptarget.cpp
  OmptCallback.cpp
  OmptTracing.cpp
  OmptTracingBuffer.cpp
  rtl.cpp
  LegacyAPI.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}

  LINK_COMPONENTS
  Support
  Object

  LINK_LIBS
  PUBLIC
  omp
)
target_include_directories(omptarget PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(omptarget PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

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

find_library(LLVM_OFFLOAD_ARCH LLVMOffloadArch HINTS ${LLVM_LIBRARY_DIR} ${LLVM_BINARY_DIR} PATH_SUFFIXES lib REQUIRED)

target_include_directories(omptarget PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})

target_link_libraries(
  omptarget
  PRIVATE hsakmt
  PRIVATE drm_amdgpu
  PRIVATE numa
  PRIVATE
  ${CMAKE_DL_LIBS}
  ${LLVM_OFFLOAD_ARCH}
  "-Wl,--no-allow-shlib-undefined")

# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
install(TARGETS omptarget LIBRARY COMPONENT omptarget DESTINATION "${RUN_PACKAGE}${OPENMP_INSTALL_LIBDIR}")
