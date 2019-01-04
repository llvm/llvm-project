#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# Try to detect in the system several dependencies required by the different
# components of libomptarget. These are the dependencies we have:
#
# libelf : required by some targets to handle the ELF files at runtime.
# libffi : required to launch target kernels given function and argument 
#          pointers.
# CUDA : required to control offloading to NVIDIA GPUs.

include (FindPackageHandleStandardArgs)

################################################################################
# Looking for libelf...
################################################################################

find_path (
  LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR
  NAMES
    libelf.h
  PATHS
    /usr/include
    /usr/local/include
    /opt/local/include
    /sw/include
    ENV CPATH
  PATH_SUFFIXES
    libelf)

find_library (
  LIBOMPTARGET_DEP_LIBELF_LIBRARIES
  NAMES
    elf
  PATHS
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    ENV LIBRARY_PATH
    ENV LD_LIBRARY_PATH)
    
set(LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS ${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR})
find_package_handle_standard_args(
  LIBOMPTARGET_DEP_LIBELF 
  DEFAULT_MSG
  LIBOMPTARGET_DEP_LIBELF_LIBRARIES
  LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS)

mark_as_advanced(
  LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS 
  LIBOMPTARGET_DEP_LIBELF_LIBRARIES)
  
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
  # Since CMake 3.3 FindCUDA.cmake defaults to using static libraries. In this
  # case CUDA_LIBRARIES contains additional linker arguments which breaks
  # get_filename_component below. Fortunately, since that change the module
  # exports CUDA_cudart_static_LIBRARY which points to a single file in the
  # right directory.
  set(cuda_library ${CUDA_LIBRARIES})
  if (DEFINED CUDA_cudart_static_LIBRARY)
    set(cuda_library ${CUDA_cudart_static_LIBRARY})
  endif()
  get_filename_component(CUDA_LIBDIR ${cuda_library} DIRECTORY)
  find_library (
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
