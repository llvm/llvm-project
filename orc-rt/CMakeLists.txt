# CMake build for ORC-RT.

#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 3.20.0)

set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

project(OrcRT LANGUAGES C CXX ASM)

list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/../runtimes/cmake/Modules"
  "${LLVM_COMMON_CMAKE_UTILS}"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

include(GNUInstallDirs)

#===============================================================================
# Setup CMake Options
#===============================================================================

option(ORC_RT_INCLUDE_DOCS "Build the ORC-RT documentation." ON)
option(ORC_RT_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(ORC_RT_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(ORC_RT_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)
option(ORC_RT_INCLUDE_TESTS "Build ORC-RT tests." ${LLVM_INCLUDE_TESTS})

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_FOLDER "orc-rt")

set(ORC_RT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ORC_RT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#===============================================================================
# Setup Source Code
#===============================================================================

if (ORC_RT_INCLUDE_DOCS)
    add_subdirectory(docs)
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(ORC_RT_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(unittests)
endif()
