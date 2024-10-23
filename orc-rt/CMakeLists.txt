# CMake build for ORC-RT.

#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 3.20.0)
set(LLVM_SUBPROJECT_TITLE "orc-rt")

set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

# Add path for custom orc-rt modules.
list(INSERT CMAKE_MODULE_PATH 0
#  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
#  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  "${LLVM_COMMON_CMAKE_UTILS}"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

set(CMAKE_FOLDER "orc-rt")

set(LIBORCRT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBORCRT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#===============================================================================
# Setup CMake Options
#===============================================================================

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

#================================
# Setup Compiler Flags
#================================

# Configure compiler. Must happen after setting the target flags.

#===============================================================================
# Setup Source Code
#===============================================================================

add_subdirectory(lib)
add_subdirectory(tools)
