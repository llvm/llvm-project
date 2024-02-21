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

if(NOT LLVM_LIBRARY_OUTPUT_INTDIR)
  set(LIBOMPTARGET_INTDIR ${LIBOMPTARGET_LIBRARY_DIR})
else()
  set(LIBOMPTARGET_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
endif()

# Message utilities.
include(LibomptargetUtils)

# Get dependencies for the different components of the project.
include(LibomptargetGetDependencies)

# LLVM source tree is required at build time for libomptarget
if (NOT LIBOMPTARGET_LLVM_INCLUDE_DIRS)
  message(FATAL_ERROR "Missing definition for LIBOMPTARGET_LLVM_INCLUDE_DIRS")
endif()

include_directories(${LIBOMPTARGET_LLVM_INCLUDE_DIRS})

# This is a list of all the targets that are supported/tested right now.
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} aarch64-unknown-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} aarch64-unknown-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} amdgcn-amd-amdhsa")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-pc-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-pc-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda-JIT-LTO")

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
else()
  option(LIBOMPTARGET_ENABLE_DEBUG "Allow debug output with the environment variable LIBOMPTARGET_DEBUG=1" OFF)
endif()
if(LIBOMPTARGET_ENABLE_DEBUG)
  add_definitions(-DOMPTARGET_DEBUG)
endif()

# No exceptions and no RTTI, except if requested.
set(offload_compile_flags -fno-exceptions)
if(NOT LLVM_ENABLE_RTTI)
  set(offload_compile_flags ${offload_compile_flags} -fno-rtti)
endif()

# TODO: Consider enabling LTO by default if supported.
# https://cmake.org/cmake/help/latest/module/CheckIPOSupported.html can be used
# to test for working LTO. However, before CMake 3.24 this will test the
# default linker and ignore options such as LLVM_ENABLE_LLD. As a result, CMake
# would test whether LTO works with the default linker but build with another one.
# In a typical scenario, libomptarget is compiled with the in-tree Clang, but
# linked with ld.gold, which requires the LLVMgold plugin, when it actually
# would work with the lld linker (or also fail because the system lld is too old
# to understand opaque pointers). Using gcc as the compiler would pass the test, but fail
# when linking with lld since does not understand gcc's LTO format.
set(LIBOMPTARGET_USE_LTO FALSE CACHE BOOL "Use LTO for the offload runtimes if available")
if (LIBOMPTARGET_USE_LTO)
  # CMake sets CMAKE_CXX_COMPILE_OPTIONS_IPO depending on the compiler and is
  # also what CheckIPOSupported uses to test support.
  list(APPEND offload_compile_flags ${CMAKE_CXX_COMPILE_OPTIONS_IPO})
  list(APPEND offload_link_flags ${CMAKE_CXX_COMPILE_OPTIONS_IPO})
endif()

# OMPT support for libomptarget
# Follow host OMPT support and check if host support has been requested.
# LIBOMP_HAVE_OMPT_SUPPORT indicates whether host OMPT support has been implemented.
# LIBOMP_OMPT_SUPPORT indicates whether host OMPT support has been requested (default is ON).
# LIBOMPTARGET_OMPT_SUPPORT indicates whether target OMPT support has been requested (default is ON).
set(OMPT_TARGET_DEFAULT FALSE)
if ((LIBOMP_HAVE_OMPT_SUPPORT) AND (LIBOMP_OMPT_SUPPORT) AND (NOT WIN32))
  set (OMPT_TARGET_DEFAULT TRUE)
endif()
set(LIBOMPTARGET_OMPT_SUPPORT ${OMPT_TARGET_DEFAULT} CACHE BOOL "OMPT-target-support?")
if ((OMPT_TARGET_DEFAULT) AND (LIBOMPTARGET_OMPT_SUPPORT))
  add_definitions(-DOMPT_SUPPORT=1)
  message(STATUS "OMPT target enabled")
else()
  set(LIBOMPTARGET_OMPT_SUPPORT FALSE)
  message(STATUS "OMPT target disabled")
endif()

pythonize_bool(LIBOMPTARGET_OMPT_SUPPORT)

# Check if this build supports the GPU libc.
set(LIBC_GPU_SUPPORT FALSE)
if("libc" IN_LIST LLVM_ENABLE_RUNTIMES AND (LIBC_GPU_BUILD OR
                                            LIBC_GPU_ARCHITECTURES))
  set(LIBC_GPU_SUPPORT TRUE)
endif()

set(LIBOMPTARGET_GPU_LIBC_SUPPORT ${LIBC_GPU_SUPPORT} CACHE BOOL
    "Libomptarget support for the GPU libc")
pythonize_bool(LIBOMPTARGET_GPU_LIBC_SUPPORT)

set(LIBOMPTARGET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
message(STATUS "OpenMP tools dir in libomptarget: ${LIBOMP_OMP_TOOLS_INCLUDE_DIR}")
include_directories(${LIBOMP_OMP_TOOLS_INCLUDE_DIR})

# Definitions for testing, for reuse when testing libomptarget-nvptx.
set(LIBOMPTARGET_OPENMP_HEADER_FOLDER "${LIBOMP_INCLUDE_DIR}" CACHE STRING
  "Path to folder containing omp.h")
set(LIBOMPTARGET_OPENMP_HOST_RTL_FOLDER "${LIBOMP_LIBRARY_DIR}" CACHE STRING
  "Path to folder containing libomp.so, and libLLVMSupport.so with profiling enabled")
set(LIBOMPTARGET_LLVM_LIBRARY_DIR "${LLVM_LIBRARY_DIR}" CACHE STRING
  "Path to folder containing llvm library libomptarget.so")
set(LIBOMPTARGET_LLVM_LIBRARY_INTDIR "${LIBOMPTARGET_INTDIR}" CACHE STRING
  "Path to folder where intermediate libraries will be output")

# Build offloading plugins and device RTLs if they are available.
add_subdirectory(plugins-nextgen)
add_subdirectory(DeviceRTL)
add_subdirectory(tools)

# Build target agnostic offloading library.
set(LIBOMPTARGET_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_subdirectory(${LIBOMPTARGET_SRC_DIR})

# Add tests.
add_subdirectory(test)

# Add unit tests if GMock/GTest is present
if (EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest)
  if (NOT TARGET llvm_gtest)
    add_subdirectory(${LLVM_THIRD_PARTY_DIR}/unittest ${CMAKE_CURRENT_BINARY_DIR}/third-party/unittest)
  endif()
  add_subdirectory(unittests)
endif()
