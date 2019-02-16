##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
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

if(OPENMP_STANDALONE_BUILD)
  # Build all libraries into a common place so that tests can find them.
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Message utilities.
include(LibomptargetUtils)

# Get dependencies for the different components of the project.
include(LibomptargetGetDependencies)

# This is a list of all the targets that are supported/tested right now.
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} aarch64-unknown-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64le-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} powerpc64-ibm-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} x86_64-pc-linux-gnu")
set (LIBOMPTARGET_ALL_TARGETS "${LIBOMPTARGET_ALL_TARGETS} nvptx64-nvidia-cuda")

# Once the plugins for the different targets are validated, they will be added to
# the list of supported targets in the current system.
set (LIBOMPTARGET_SYSTEM_TARGETS "")

# If building this library in debug mode, we define a macro to enable
# dumping progress messages at runtime.
string( TOLOWER "${CMAKE_BUILD_TYPE}" LIBOMPTARGET_CMAKE_BUILD_TYPE)
if(LIBOMPTARGET_CMAKE_BUILD_TYPE MATCHES debug)
  add_definitions(-DOMPTARGET_DEBUG)
  add_definitions(-g)
  add_definitions(-O0)
endif()

include_directories(include)

# Build target agnostic offloading library.
add_subdirectory(src)

# Retrieve the path to the resulting library so that it can be used for 
# testing.
get_target_property(LIBOMPTARGET_LIBRARY_DIR omptarget LIBRARY_OUTPUT_DIRECTORY)
if(NOT LIBOMPTARGET_LIBRARY_DIR)
  set(LIBOMPTARGET_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Definitions for testing, for reuse when testing libomptarget-nvptx.
if(OPENMP_STANDALONE_BUILD)
  set(LIBOMPTARGET_OPENMP_HEADER_FOLDER "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src" CACHE STRING
    "Path to folder containing omp.h")
  set(LIBOMPTARGET_OPENMP_HOST_RTL_FOLDER "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src" CACHE STRING
    "Path to folder containing libomp.so")
else()
  set(LIBOMPTARGET_OPENMP_HEADER_FOLDER "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src")
endif()


# Build offloading plugins and device RTLs if they are available.
add_subdirectory(plugins)
add_subdirectory(deviceRTLs)

# Add tests.
add_subdirectory(test)
