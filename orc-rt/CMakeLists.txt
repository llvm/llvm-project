# CMake build for ORC-RT.

#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 3.20.0)

set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

project(OrcRT LANGUAGES C CXX ASM)

include(GNUInstallDirs)

#===============================================================================
# Setup CMake Options
#===============================================================================

option(ORC_RT_INCLUDE_DOCS "Build the ORC-RT documentation." ON)
option(ORC_RT_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(ORC_RT_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(ORC_RT_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_FOLDER "orc-rt")

#===============================================================================
# Setup Source Code
#===============================================================================

if (ORC_RT_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
