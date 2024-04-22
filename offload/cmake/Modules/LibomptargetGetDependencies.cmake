#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

# Try to detect in the system several dependencies required by the different
# components of libomptarget. These are the dependencies we have:
#
# libffi : required to launch target kernels given function and argument
#          pointers.
# CUDA : required to control offloading to NVIDIA GPUs.

include (FindPackageHandleStandardArgs)

################################################################################
# Looking for LLVM...
################################################################################

if (OPENMP_STANDALONE_BUILD)
  # Complete LLVM package is required for building libomptarget
  # in an out-of-tree mode.
  find_package(LLVM REQUIRED)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVM in: ${LLVM_DIR}")
  list(APPEND LIBOMPTARGET_LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
  list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
  include(AddLLVM)
  if(TARGET omptarget)
    message(FATAL_ERROR "CMake target 'omptarget' already exists. "
                        "Use an LLVM installation that doesn't expose its 'omptarget' target.")
  endif()
else()
  # Note that OPENMP_STANDALONE_BUILD is FALSE, when
  # openmp is built with -DLLVM_ENABLE_RUNTIMES="openmp" vs
  # -DLLVM_ENABLE_PROJECTS="openmp", but openmp build
  # is actually done as a standalone project build with many
  # LLVM CMake variables propagated to it.
  list(APPEND LIBOMPTARGET_LLVM_INCLUDE_DIRS
    ${LLVM_MAIN_INCLUDE_DIR} ${LLVM_BINARY_DIR}/include
    )
  message(STATUS
    "Using LLVM include directories: ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}")
endif()

################################################################################
# Looking for libffi...
################################################################################
find_package(FFI QUIET)
set(LIBOMPTARGET_DEP_LIBFFI_FOUND ${FFI_FOUND})

################################################################################
# Looking for CUDA...
################################################################################

find_package(CUDAToolkit QUIET)
set(LIBOMPTARGET_DEP_CUDA_FOUND ${CUDAToolkit_FOUND})

################################################################################
# Looking for NVIDIA GPUs...
################################################################################
set(LIBOMPTARGET_DEP_CUDA_ARCH "sm_35")

if(TARGET nvptx-arch)
  get_property(LIBOMPTARGET_NVPTX_ARCH TARGET nvptx-arch PROPERTY LOCATION)
else()
  find_program(LIBOMPTARGET_NVPTX_ARCH NAMES nvptx-arch
               PATHS ${LLVM_TOOLS_BINARY_DIR}/bin)
endif()

if(LIBOMPTARGET_NVPTX_ARCH)
  execute_process(COMMAND ${LIBOMPTARGET_NVPTX_ARCH}
                  OUTPUT_VARIABLE LIBOMPTARGET_NVPTX_ARCH_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" ";" nvptx_arch_list "${LIBOMPTARGET_NVPTX_ARCH_OUTPUT}")
  if(nvptx_arch_list)
    set(LIBOMPTARGET_FOUND_NVIDIA_GPU TRUE)
    set(LIBOMPTARGET_NVPTX_DETECTED_ARCH_LIST "${nvptx_arch_list}")
    list(GET nvptx_arch_list 0 LIBOMPTARGET_DEP_CUDA_ARCH)
  endif()
endif()


################################################################################
# Looking for AMD GPUs...
################################################################################

if(TARGET amdgpu-arch)
  get_property(LIBOMPTARGET_AMDGPU_ARCH TARGET amdgpu-arch PROPERTY LOCATION)
else()
  find_program(LIBOMPTARGET_AMDGPU_ARCH NAMES amdgpu-arch
               PATHS ${LLVM_TOOLS_BINARY_DIR}/bin)
endif()

if(LIBOMPTARGET_AMDGPU_ARCH)
  execute_process(COMMAND ${LIBOMPTARGET_AMDGPU_ARCH}
                  OUTPUT_VARIABLE LIBOMPTARGET_AMDGPU_ARCH_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" ";" amdgpu_arch_list "${LIBOMPTARGET_AMDGPU_ARCH_OUTPUT}")
  if(amdgpu_arch_list)
    set(LIBOMPTARGET_FOUND_AMDGPU_GPU TRUE)
    set(LIBOMPTARGET_AMDGPU_DETECTED_ARCH_LIST "${amdgpu_arch_list}")
  endif()
endif()

set(OPENMP_PTHREAD_LIB ${LLVM_PTHREAD_LIB})
