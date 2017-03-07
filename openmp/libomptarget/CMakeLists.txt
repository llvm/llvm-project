##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
# 
##===----------------------------------------------------------------------===##
#
# Build offloading library libomptarget.so.
#
##===----------------------------------------------------------------------===##

# CMAKE libomptarget
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Add cmake directory to search for custom cmake functions.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

# Standalone build or part of LLVM?
set(LIBOMPTARGET_STANDALONE_BUILD FALSE)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}" OR
   "${CMAKE_SOURCE_DIR}/libomptarget" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  project(libomptarget C CXX)
  set(LIBOMPTARGET_STANDALONE_BUILD TRUE)
endif()
  
  
if(${LIBOMPTARGET_STANDALONE_BUILD})
  set(LIBOMPTARGET_ENABLE_WERROR FALSE CACHE BOOL
    "Enable -Werror flags to turn warnings into errors for supporting compilers.")
  # CMAKE_BUILD_TYPE was not defined, set default to Release
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  set(LIBOMPTARGET_LIBDIR_SUFFIX "" CACHE STRING
    "suffix of lib installation directory, e.g. 64 => lib64")
else()
  set(LIBOMPTARGET_ENABLE_WERROR ${LLVM_ENABLE_WERROR})
  # If building in tree, we honor the same install suffix LLVM uses.
  set(LIBOMPTARGET_LIBDIR_SUFFIX ${LLVM_LIBDIR_SUFFIX})
endif()

# Compiler flag checks.
include(config-ix)

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

# Set base directories - required for lit to locate the tests.
set(LIBOMPTARGET_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBOMPTARGET_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# We need C++11 support.
if(LIBOMPTARGET_HAVE_STD_CPP11_FLAG)
  
  libomptarget_say("Building offloading runtime library libomptarget.")

  # Enable support for C++11.
  add_definitions(-std=c++11)
  
  if(LIBOMPTARGET_ENABLE_WERROR AND LIBOMPTARGET_HAVE_WERROR_FLAG)
    add_definitions(-Werror)
  endif()
  
  # If building this library in debug mode, we define a macro to enable
  # dumping progress messages at runtime.
  string( TOLOWER "${CMAKE_BUILD_TYPE}" LIBOMPTARGET_CMAKE_BUILD_TYPE)
  if(LIBOMPTARGET_CMAKE_BUILD_TYPE MATCHES debug)
    add_definitions(-DOMPTARGET_DEBUG)
    add_definitions(-g)
    add_definitions(-O0)
  endif()
  
  set(src_files
    src/omptarget.cpp
  )
  
  include_directories(src/)
  
  # Build libomptarget library with libdl dependency.
  add_library(omptarget SHARED ${src_files})
  target_link_libraries(omptarget
    dl
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
  
  # Install libomptarget under the lib destination folder.
  install(TARGETS omptarget LIBRARY DESTINATION lib${LIBOMPTARGET_LIBDIR_SUFFIX})
  
  # Retrieve the path to the resulting library so that it can be used for 
  # testing.
  get_target_property(LIBOMPTARGET_LIBRARY_DIR omptarget LIBRARY_OUTPUT_DIRECTORY)
  if(NOT LIBOMPTARGET_LIBRARY_DIR)
    set(LIBOMPTARGET_LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  # Build offloading plugins and device RTLs if they are available.
  add_subdirectory(plugins)
  
  # Add tests.
  add_subdirectory(test)
  
else(LIBOMPTARGET_HAVE_STD_CPP11_FLAG)
  libomptarget_say("Not building offloading runtime library libomptarget: host compiler must have c++11 support.")
endif(LIBOMPTARGET_HAVE_STD_CPP11_FLAG)
