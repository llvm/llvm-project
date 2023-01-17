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
# VEOS : required to control offloading to NEC Aurora.

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
find_package(PkgConfig)

pkg_check_modules(LIBOMPTARGET_SEARCH_LIBFFI QUIET libffi)

find_path (
  LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR
  NAMES
    ffi.h
  HINTS
    ${LIBOMPTARGET_SEARCH_LIBFFI_INCLUDEDIR}
    ${LIBOMPTARGET_SEARCH_LIBFFI_INCLUDE_DIRS}
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    /sw/include
    ENV CPATH)

# Don't bother look for the library if the header files were not found.
if (LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR)
  find_library (
      LIBOMPTARGET_DEP_LIBFFI_LIBRARIES
    NAMES
      ffi
    HINTS
      ${LIBOMPTARGET_SEARCH_LIBFFI_LIBDIR}
      ${LIBOMPTARGET_SEARCH_LIBFFI_LIBRARY_DIRS}
    PATHS
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib
      ENV LIBRARY_PATH
      ENV LD_LIBRARY_PATH)
endif()

set(LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS ${LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR})
find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBFFI
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBFFI_LIBRARIES
  LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIRS
  LIBOMPTARGET_DEP_LIBFFI_LIBRARIES)

################################################################################
# Looking for CUDA...
################################################################################
if (CUDA_TOOLKIT_ROOT_DIR)
  set(LIBOMPTARGET_CUDA_TOOLKIT_ROOT_DIR_PRESET TRUE)
endif()
find_package(CUDA QUIET)

# Identify any locally installed GPUs to use for testing.
set(LIBOMPTARGET_DEP_CUDA_ARCH "sm_35")

find_program(LIBOMPTARGET_NVPTX_ARCH NAMES nvptx-arch PATHS ${LLVM_BINARY_DIR}/bin)
if(LIBOMPTARGET_NVPTX_ARCH)
  execute_process(COMMAND ${LIBOMPTARGET_NVPTX_ARCH}
                  OUTPUT_VARIABLE LIBOMPTARGET_NVPTX_ARCH_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(FIND "${LIBOMPTARGET_NVPTX_ARCH_OUTPUT}" "\n" first_arch_string)
  string(SUBSTRING "${LIBOMPTARGET_NVPTX_ARCH_OUTPUT}" 0 ${first_arch_string}
         arch_string)
  if(arch_string)
    set(LIBOMPTARGET_DEP_CUDA_ARCH "${arch_string}")
  endif()
endif()

set(LIBOMPTARGET_DEP_CUDA_FOUND ${CUDA_FOUND})
set(LIBOMPTARGET_DEP_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

mark_as_advanced(
  LIBOMPTARGET_DEP_CUDA_FOUND
  LIBOMPTARGET_DEP_CUDA_INCLUDE_DIRS)

################################################################################
# Looking for CUDA Driver API... (needed for CUDA plugin)
################################################################################

find_library (
    LIBOMPTARGET_DEP_CUDA_DRIVER_LIBRARIES
  NAMES
    cuda
  PATHS
    /lib64)

# There is a libcuda.so in lib64/stubs that can be used for linking.
if (NOT LIBOMPTARGET_DEP_CUDA_DRIVER_LIBRARIES AND CUDA_FOUND)
  get_filename_component(CUDA_LIBDIR "${CUDA_cudart_static_LIBRARY}" DIRECTORY)
  find_library(
      LIBOMPTARGET_DEP_CUDA_DRIVER_LIBRARIES
    NAMES
      cuda
    HINTS
      "${CUDA_LIBDIR}/stubs")
endif()

find_package_handle_standard_args(
  LIBOMPTARGET_DEP_CUDA_DRIVER
  DEFAULT_MSG
  LIBOMPTARGET_DEP_CUDA_DRIVER_LIBRARIES)

mark_as_advanced(LIBOMPTARGET_DEP_CUDA_DRIVER_LIBRARIES)

################################################################################
# Looking for VEO...
################################################################################

find_path (
  LIBOMPTARGET_DEP_VEO_INCLUDE_DIR
  NAMES
    ve_offload.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    /sw/include
    /opt/nec/ve/veos/include
    ENV CPATH
  PATH_SUFFIXES
    libveo)

find_library (
  LIBOMPTARGET_DEP_VEO_LIBRARIES
  NAMES
    veo
  PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    /opt/nec/ve/veos/lib64
    ENV LIBRARY_PATH
    ENV LD_LIBRARY_PATH)

find_library(
  LIBOMPTARGET_DEP_VEOSINFO_LIBRARIES
  NAMES
    veosinfo
  PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    /opt/nec/ve/veos/lib64
    ENV LIBRARY_PATH
    ENV LD_LIBRARY_PATH)

set(LIBOMPTARGET_DEP_VEO_INCLUDE_DIRS ${LIBOMPTARGET_DEP_VEO_INCLUDE_DIR})
find_package_handle_standard_args(
  LIBOMPTARGET_DEP_VEO
  DEFAULT_MSG
  LIBOMPTARGET_DEP_VEO_LIBRARIES
  LIBOMPTARGET_DEP_VEOSINFO_LIBRARIES
  LIBOMPTARGET_DEP_VEO_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_VEO_FOUND
  LIBOMPTARGET_DEP_VEO_INCLUDE_DIRS)

# Looking for CUDA libdevice subdirectory
#
# Special case for Debian/Ubuntu to have nvidia-cuda-toolkit work
# out of the box. More info on http://bugs.debian.org/882505
################################################################################

set(LIBOMPTARGET_CUDA_LIBDEVICE_SUBDIR nvvm/libdevice)

# Don't alter CUDA_TOOLKIT_ROOT_DIR if the user specified it, if a value was
# already cached for it, or if it already has libdevice.  Otherwise, on
# Debian/Ubuntu, look where the nvidia-cuda-toolkit package normally installs
# libdevice.
if (NOT LIBOMPTARGET_CUDA_TOOLKIT_ROOT_DIR_PRESET AND
    NOT EXISTS
      "${CUDA_TOOLKIT_ROOT_DIR}/${LIBOMPTARGET_CUDA_LIBDEVICE_SUBDIR}")
  find_program(LSB_RELEASE lsb_release)
  if (LSB_RELEASE)
    execute_process(COMMAND ${LSB_RELEASE} -is
      OUTPUT_VARIABLE LSB_RELEASE_ID
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(candidate_dir /usr/lib/cuda)
    if ((LSB_RELEASE_ID STREQUAL "Debian" OR LSB_RELEASE_ID STREQUAL "Ubuntu")
        AND EXISTS "${candidate_dir}/${LIBOMPTARGET_CUDA_LIBDEVICE_SUBDIR}")
      set(CUDA_TOOLKIT_ROOT_DIR "${candidate_dir}" CACHE PATH
          "Toolkit location." FORCE)
    endif()
  endif()
endif()

set(OPENMP_PTHREAD_LIB ${LLVM_PTHREAD_LIB})
