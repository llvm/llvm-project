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
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} aarch64-unknown-linux-gnu-oldDriver")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} aarch64-unknown-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} amdgcn-amd-amdhsa")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu-oldDriver")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu-oldDriver")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-pc-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-pc-linux-gnu-oldDriver")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-pc-linux-gnu-LTO")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda-oldDriver")
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

# No exceptions and no RTTI, except if requested.
set(offload_compile_flags -fno-exceptions)
if(NOT LLVM_ENABLE_RTTI)
  set(offload_compile_flags ${offload_compile_flags} -fno-rtti)
endif()

# If LTO is not explicitly disabled we check if we can enable it and do so.
set(LIBOMPTARGET_USE_LTO TRUE CACHE BOOL "Use LTO for the offload runtimes if available")
if (LIBOMPTARGET_USE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT use_lto OUTPUT output)
  if(use_lto)
    set(offload_compile_flags ${offload_compile_flags} -flto)
    set(offload_link_flags ${offload_link_flags} -flto)
  else()
    message(WARNING "LTO is not supported: ${output}")
  endif()
endif()

# OMPT support for libomptarget
# Follow host OMPT support and check if host support has been requested.
# LIBOMP_HAVE_OMPT_SUPPORT indicates whether host OMPT support has been implemented.
# LIBOMP_OMPT_SUPPORT indicates whether host OMPT support has been requested (default is ON).
# LIBOMPTARGET_OMPT_SUPPORT indicates whether target OMPT support has been requested (default is ON).
set(OMPT_TARGET_DEFAULT FALSE)
if ((LIBOMP_HAVE_OMPT_SUPPORT) AND (NOT WIN32))
  set (OMPT_TARGET_DEFAULT TRUE)
endif()
set(LIBOMPTARGET_OMPT_SUPPORT ${OMPT_TARGET_DEFAULT} CACHE BOOL "OMPT-target-support?")
if (LIBOMPTARGET_OMPT_SUPPORT)
  add_definitions(-DOMPT_SUPPORT=1)
  message(STATUS "OMPT target enabled")
else()
  set(LIBOMPTARGET_OMPT_SUPPORT FALSE)
  message(STATUS "OMPT target disabled")
endif()

include_directories(include)
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

find_path (
  LIBOMP_OMP_TOOLS_INCLUDE_DIR
  NAMES
    omp-tools.h
  PATHS
    ${CMAKE_BINARY_DIR}/openmp/runtime/src
    ${CMAKE_BINARY_DIR}/runtime/src
  )

include_directories(${LIBOMPTARGET_INCLUDE_DIR} ${LIBOMP_OMP_TOOLS_INCLUDE_DIR})

# Various LLVM_TOOLS are needed to build libomptarget
find_package(LLVM QUIET CONFIG PATHS
   ${LLVM_INSTALL_PREFIX}
   ${LIBOMPTARGET_NVPTX_CUDA_COMPILER_DIR}
   ${LIBOMPTARGET_NVPTX_CUDA_LINKER_DIR}
   ${CMAKE_CXX_COMPILER_DIR}
   NO_DEFAULT_PATH)
if(LLVM_DIR)
  message(" -- LLVM found at ${LLVM_DIR}")
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(CLANG_OFFLOAD_BUNDLER_TOOL clang-offload-bundler PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(AR_TOOL llvm-ar PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(LINK_TOOL llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  find_program(OPT_TOOL opt PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING AND NOT OPENMP_STANDALONE_BUILD)
  # LLVM in-tree builds may use CMake target names to discover the tools.
  set(CLANG_TOOL $<TARGET_FILE:clang>)
  set(CLANG_OFFLOAD_BUNDLER_TOOL $<TARGET_FILE:clang-offload-bundler>)
  set(AR_TOOL $<TARGET_FILE:llvm-ar>)
  set(LINK_TOOL $<TARGET_FILE:llvm-link>)
  set(OPT_TOOL $<TARGET_FILE:opt>)
else()
  message(" ====== WARNING! no LLVM found! some libomptarget components may be skipped")
endif()

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
# Currently device libm functions are created by clang headers.  These do not
# work for FORTRAN. The libm here  builds a libm device library by turning on
# __BUILD_MATH_BUILTINS_LIB__ This converts the math functions in the clang
# headers from static to extern to build device linkable libm for FORTRAN.
# Eventually, we can make a linkable device libm for c and c++ and remove a
# lot of header definitions that get compiled with every offload compilation.
add_subdirectory(libm)

add_subdirectory(hostexec)
add_subdirectory(tools)
set(PREP_TOOL $<TARGET_FILE:prep-libomptarget-bc>)
add_subdirectory(DeviceRTL)

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
