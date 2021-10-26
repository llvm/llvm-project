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

include_directories(
  ${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS}
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
)

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

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.cuda LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")

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
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} nvptx64-nvidia-cuda nvptx64-nvidia-cuda-newRTL" PARENT_SCOPE)
else()
  libomptarget_say("Disabling tests using CUDA plugin as cuda may not be available")
endif()
