# CMake build for ORC-RT.

#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 3.20.0)

set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

project(LibOrcRT LANGUAGES C CXX ASM)

include(GNUInstallDirs)

#===============================================================================
# Setup CMake Options
#===============================================================================

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_FOLDER "orc-rt")

#===============================================================================
# Setup Source Code
#===============================================================================

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
