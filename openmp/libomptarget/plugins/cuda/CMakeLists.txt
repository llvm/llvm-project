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
  libomptarget_say("Not building CUDA offloading plugin: LIBOMPTARGET_BUILD_CUDA_PLUGIN is false")
  return()
endif()

if (NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building CUDA offloading plugin: only support CUDA in Linux x86_64, ppc64le, or aarch64 hosts.")
  return()
elseif (NOT LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building CUDA offloading plugin: libelf dependency not found.")
  return()
endif()

libomptarget_say("Building CUDA offloading plugin.")

# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=CUDA)

set(LIBOMPTARGET_DLOPEN_LIBCUDA OFF)
option(LIBOMPTARGET_FORCE_DLOPEN_LIBCUDA "Build with dlopened libcuda" ${LIBOMPTARGET_DLOPEN_LIBCUDA})

set(LIBOMPTARGET_CAN_LINK_LIBCUDA FALSE)
if (LIBOMPTARGET_DEP_CUDA_FOUND AND LIBOMPTARGET_DEP_CUDA_DRIVER_FOUND)
  set(LIBOMPTARGET_CAN_LINK_LIBCUDA TRUE)
endif()

if (LIBOMPTARGET_CAN_LINK_LIBCUDA AND NOT LIBOMPTARGET_FORCE_DLOPEN_LIBCUDA)
  libomptarget_say("Building CUDA plugin linked against libcuda")
  include_directories(${LIBOMPTARGET_DEP_CUDA_INCLUDE_DIRS})
  add_library(omptarget.rtl.cuda SHARED src/rtl.cpp)
  set (LIBOMPTARGET_DEP_LIBRARIES ${LIBOMPTARGET_DEP_CUDA_DRIVER_LIBRARIES})
else()
  libomptarget_say("Building CUDA plugin for dlopened libcuda")
  include_directories(dynamic_cuda)
  add_library(omptarget.rtl.cuda SHARED src/rtl.cpp dynamic_cuda/cuda.cpp)
  set (LIBOMPTARGET_DEP_LIBRARIES ${CMAKE_DL_LIBS})
endif()
add_dependencies(omptarget.rtl.cuda omptarget.devicertl.nvptx)

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.cuda LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")

target_include_directories(omptarget.rtl.cuda PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS}
)

target_link_libraries(omptarget.rtl.cuda
  elf_common
  MemoryManager
  ${LIBOMPTARGET_DEP_LIBRARIES}
  ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
  ${OPENMP_PTHREAD_LIB}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
  "-Wl,-z,defs")

# Report to the parent scope that we are building a plugin for CUDA.
# This controls whether tests are run for the nvptx offloading target
# Run them if libcuda is available, or if the user explicitly asked for dlopen
# Otherwise this plugin is being built speculatively and there may be no cuda available
if (LIBOMPTARGET_CAN_LINK_LIBCUDA OR LIBOMPTARGET_FORCE_DLOPEN_LIBCUDA)
  libomptarget_say("Enable tests using CUDA plugin")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} nvptx64-nvidia-cuda nvptx64-nvidia-cuda-oldDriver" PARENT_SCOPE)
  list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.cuda")
  set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
else()
  libomptarget_say("Disabling tests using CUDA plugin as cuda may not be available")
endif()
