# CMake build for ORC-RT.

#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 3.20.0)

option(LLVM_RT_INCLUDE_TESTS "Build orcrt tests." ${LLVM_INCLUDE_TESTS})


set(LLVM_COMMON_CMAKE_UTILS ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

if (ORC_RT_INCLUDE_TESTS)
    set(LLVM_INCLUDE_UTILS YES)
endif()

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

if(LLVM_INCLUDE_TESTS)
    if(NOT TARGET FileCheck)
        find_program(FILECHECK_EXE
    NAMES FileCheck
    HINTS ${LLVM_RT_TOOLS_BINARY_DIR}
    DOC "FileCheck executable")
    endif()
    if(FILECHECK_EXE)
        add_executable(FileCheck IMPORTED GLOBAL)
        set_property(TARGET FileCheck PROPERTY IMPORTED_LOCATION "${FILECHECK_EXE}")
        message(DEBUG "Using FileCheck: ${FILECHECK_EXE}")
    endif()
    if(NOT TARGET count)
        find_program(COUNT_EXE
            NAMES count
            HINTS ${LLVM_RT_TOOLS_BINARY_DIR}
            DOC "count executable")
    endif()
    if(COUNT_EXE)
        add_executable(count IMPORTED GLOBAL)
        set_property(TARGET count PROPERTY IMPORTED_LOCATION "${COUNT_EXE}")
        message(DEBUG "Using count: ${COUNT_EXE}")
    endif()
    if(NOT TARGET not)
        find_program(NOT_EXE
            NAMES not
            HINTS ${LLVM_RT_TOOLS_BINARY_DIR}
            DOC "not executable")
    endif()
    if(NOT_EXE)
        add_executable(not IMPORTED GLOBAL)
        set_property(TARGET not PROPERTY IMPORTED_LOCATION "${NOT_EXE}")
        message(DEBUG "Using not: ${NOT_EXE}")
    endif()

    set(ORCRT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(ORCRT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

  add_subdirectory(test)
  set(UNITTEST_DIR ${LLVM_THIRD_PARTY_DIR}/unittest)
  if(EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h)
    if (NOT TARGET gtest)
      add_subdirectory(${UNITTEST_DIR} third-party/unittest)
    endif()
     add_subdirectory(unittests)
  endif()
endif()