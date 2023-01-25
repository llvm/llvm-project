##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a CUDA machine if available.
#
##===----------------------------------------------------------------------===##
set(LIBOMPTARGET_BUILD_CUDA_PLUGIN TRUE CACHE BOOL
  "Whether to build CUDA plugin")
if (NOT LIBOMPTARGET_BUILD_CUDA_PLUGIN)
  libomptarget_say("Not building CUDA NextGen offloading plugin: LIBOMPTARGET_BUILD_CUDA_PLUGIN is false")
  return()
endif()

if (NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building CUDA NextGen offloading plugin: only support CUDA in Linux x86_64, ppc64le, or aarch64 hosts.")
  return()
endif()

libomptarget_say("Building CUDA NextGen offloading plugin.")

set(LIBOMPTARGET_DLOPEN_LIBCUDA OFF)
option(LIBOMPTARGET_FORCE_DLOPEN_LIBCUDA "Build with dlopened libcuda" ${LIBOMPTARGET_DLOPEN_LIBCUDA})

add_llvm_library(omptarget.rtl.cuda.nextgen SHARED
  src/rtl.cpp

  LINK_COMPONENTS
  Support
  Object

  LINK_LIBS PRIVATE
  elf_common
  MemoryManager
  PluginInterface
  ${OPENMP_PTHREAD_LIB}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports,-z,defs"

  NO_INSTALL_RPATH
)

if(LIBOMPTARGET_DEP_CUDA_FOUND AND NOT LIBOMPTARGET_FORCE_DLOPEN_LIBCUDA)
  libomptarget_say("Building CUDA plugin linked against libcuda")
  target_link_libraries(omptarget.rtl.cuda.nextgen PRIVATE CUDA::cuda_driver)
else()
  libomptarget_say("Building CUDA plugin for dlopened libcuda")
  target_include_directories(omptarget.rtl.cuda.nextgen PRIVATE dynamic_cuda)
  target_sources(omptarget.rtl.cuda.nextgen PRIVATE dynamic_cuda/cuda.cpp)
endif()
add_dependencies(omptarget.rtl.cuda.nextgen omptarget.devicertl.nvptx)

# Define debug prefix. TODO: This should be automatized in the Debug.h but it
# requires changing the original plugins.
target_compile_definitions(omptarget.rtl.cuda.nextgen PRIVATE TARGET_NAME="CUDA")
target_compile_definitions(omptarget.rtl.cuda.nextgen PRIVATE DEBUG_PREFIX="TARGET CUDA RTL")

target_include_directories(omptarget.rtl.cuda.nextgen PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.cuda.nextgen LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
set_target_properties(omptarget.rtl.cuda.nextgen PROPERTIES
  INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_VISIBILITY_PRESET protected)
