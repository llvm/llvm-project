##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build offloading library and related plugins.
#
##===----------------------------------------------------------------------===##

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  message(FATAL_ERROR "Direct configuration not supported, please use parent directory!")
endif()

# Add cmake directory to search for custom cmake functions.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

# Set the path of all resulting libraries to a unified location so that it can
# be used for testing.
set(LIBOMPTARGET_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBOMPTARGET_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBOMPTARGET_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBOMPTARGET_LIBRARY_DIR})

# Message utilities.
include(LibomptargetUtils)

# Get dependencies for the different components of the project.
include(LibomptargetGetDependencies)

# LLVM source tree is required at build time for libomptarget
if (NOT LIBOMPTARGET_LLVM_INCLUDE_DIRS)
  message(FATAL_ERROR "Missing definition for LIBOMPTARGET_LLVM_INCLUDE_DIRS")
endif()

# This is a list of all the targets that are supported/tested right now.
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} aarch64-unknown-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} amdgcn-amd-amdhsa")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} amdgcn-amd-amdhsa-newRTL")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-pc-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda-newRTL")

# Once the plugins for the different targets are validated, they will be added to
# the list of supported targets in the current system.
set (LIBOMPTARGET_SYSTEM_TARGETS "")
set (LIBOMPTARGET_TESTED_PLUGINS "")

# Check whether using debug mode. In debug mode, allow dumping progress
# messages at runtime by default. Otherwise, it can be enabled
# independently using the LIBOMPTARGET_ENABLE_DEBUG option.
string( TOLOWER "${CMAKE_BUILD_TYPE}" LIBOMPTARGET_CMAKE_BUILD_TYPE)
if(LIBOMPTARGET_CMAKE_BUILD_TYPE MATCHES debug)
  option(LIBOMPTARGET_ENABLE_DEBUG "Allow debug output with the environment variable LIBOMPTARGET_DEBUG=1" ON)
  add_definitions(-DDEBUG)
else()
  option(LIBOMPTARGET_ENABLE_DEBUG "Allow debug output with the environment variable LIBOMPTARGET_DEBUG=1" OFF)
endif()
if(LIBOMPTARGET_ENABLE_DEBUG)
  add_definitions(-DOMPTARGET_DEBUG)
endif()

# OMPD support for libomptarget (currently only with cuda)
set(LIBOMPTARGET_OMPD_SUPPORT FALSE CACHE BOOL "OMPD-support?")
if (LIBOMPTARGET_OMPD_SUPPORT)
  add_definitions(-DOMPD_SUPPORT=1)
endif()

# OMPT support for libomptarget
set(OMPT_TARGET_DEFAULT FALSE)
if ((LIBOMP_HAVE_OMPT_SUPPORT) AND (NOT WIN32))
  set (OMPT_TARGET_DEFAULT TRUE)
endif()
set(LIBOMPTARGET_OMPT_SUPPORT ${OMPT_TARGET_DEFAULT} CACHE BOOL "OMPT-target-support?")
if (LIBOMPTARGET_OMPT_SUPPORT)
  add_definitions(-DOMPT_SUPPORT=1)
  message(STATUS "OMPT target enabled")
else()
  message(STATUS "OMPT target disabled")
endif()

include_directories(include)
pythonize_bool(LIBOMPTARGET_OMPT_SUPPORT)

set(LIBOMPTARGET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_path (
  LIBOMP_OMP_TOOLS_INCLUDE_DIR
  NAMES
    omp-tools.h
  PATHS
    ${CMAKE_BINARY_DIR}/openmp/runtime/src
    ${CMAKE_BINARY_DIR}/runtime/src
  )

include_directories(${LIBOMPTARGET_INCLUDE_DIR} ${LIBOMP_OMP_TOOLS_INCLUDE_DIR})

# Build target agnostic offloading library.
set(LIBOMPTARGET_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${LIBOMPTARGET_SRC_DIR})

# Definitions for testing, for reuse when testing libomptarget-nvptx.
set(LIBOMPTARGET_OPENMP_HEADER_FOLDER "${LIBOMP_INCLUDE_DIR}" CACHE STRING
  "Path to folder containing omp.h")
set(LIBOMPTARGET_OPENMP_HOST_RTL_FOLDER "${LIBOMP_LIBRARY_DIR}" CACHE STRING
  "Path to folder containing libomp.so, and libLLVMSupport.so with profiling enabled")


# Build offloading plugins and device RTLs if they are available.
add_subdirectory(plugins)
add_subdirectory(deviceRTLs)
add_subdirectory(hostrpc)
add_subdirectory(DeviceRTL)
add_subdirectory(tools)

# Add tests.
add_subdirectory(test)
