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

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  message(FATAL_ERROR "Direct configuration not supported, please use parent directory!")
endif()

# Add cmake directory to search for custom cmake functions.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})

if(OPENMP_STANDALONE_BUILD)
  # Build all libraries into a common place so that tests can find them.
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

# Set base directories - required for lit to locate the tests.
set(LIBOMPTARGET_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBOMPTARGET_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

libomptarget_say("Building offloading runtime library libomptarget.")

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
  ${CMAKE_DL_LIBS}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")

# Install libomptarget under the lib destination folder.
install(TARGETS omptarget LIBRARY DESTINATION lib${OPENMP_LIBDIR_SUFFIX})

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
