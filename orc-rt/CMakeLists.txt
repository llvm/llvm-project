# CMake build for ORC-RT.

#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 3.20.0)
if("${CMAKE_VERSION}" VERSION_LESS "3.31.0")
  message(WARNING
    "Your CMake version is ${CMAKE_VERSION}. Starting with LLVM 24, the "
    "minimum version of CMake required to build LLVM will become 3.31.0, and "
    "using an older CMake will become an error. Please upgrade your CMake to "
    "at least 3.31.0 now to avoid issues in the future.")
endif()

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

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# --- Project variables ---
set(CMAKE_FOLDER "orc-rt")
set(ORC_RT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ORC_RT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#===============================================================================
# Setup CMake Options
#===============================================================================

# --- Build components ---
option(ORC_RT_INCLUDE_DOCS "Build the ORC-RT documentation." ON)
option(ORC_RT_INCLUDE_TESTS "Build ORC-RT tests." ${LLVM_INCLUDE_TESTS})

# --- Build behavior ---
option(ORC_RT_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(ORC_RT_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(ORC_RT_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)

# --- Language runtime (RTTI / exceptions) ---
option(ORC_RT_ENABLE_RTTI "Enable RTTI." ON)
option(ORC_RT_ENABLE_EXCEPTIONS "Enable exceptions." ON)

# Derive the compile flags implied by the language-runtime options.
set(ORC_RT_COMPILE_FLAGS)
if(ORC_RT_ENABLE_RTTI)
  list(APPEND ORC_RT_COMPILE_FLAGS -frtti)
else()
  list(APPEND ORC_RT_COMPILE_FLAGS -fno-rtti)
endif()

if(ORC_RT_ENABLE_EXCEPTIONS)
  list(APPEND ORC_RT_COMPILE_FLAGS -fexceptions)
else()
  list(APPEND ORC_RT_COMPILE_FLAGS -fno-exceptions)
endif()

# --- Logging ---
# User-settable options (everything below them is validation and derivation):
#   ORC_RT_LOG_BACKEND  none (default) | printf | os_log
#   ORC_RT_LOG_LEVEL    error | warning | info (default) | debug
set(orc_rt_log_level_default "info")
set(ORC_RT_LOG_LEVEL "${orc_rt_log_level_default}" CACHE STRING
  "Lowest log level compiled in: error, warning, info, or debug")
set(orc_rt_log_levels error warning info debug)
set_property(CACHE ORC_RT_LOG_LEVEL PROPERTY STRINGS ${orc_rt_log_levels})

set(ORC_RT_LOG_BACKEND "none" CACHE STRING
  "ORC-RT logging backend: none, printf, os_log")
set(orc_rt_log_backends none printf os_log)
set_property(CACHE ORC_RT_LOG_BACKEND PROPERTY STRINGS ${orc_rt_log_backends})

# Validate the level and derive its config.h macro symbol.
if(NOT ORC_RT_LOG_LEVEL IN_LIST orc_rt_log_levels)
  message(FATAL_ERROR "ORC_RT_LOG_LEVEL must be one of: ${orc_rt_log_levels}")
endif()
string(TOUPPER "ORC_RT_LOG_LEVEL_${ORC_RT_LOG_LEVEL}" ORC_RT_LOG_LEVEL_VALUE)

# Validate the backend (os_log is Apple-only) and derive its config.h macro symbol.
if(NOT ORC_RT_LOG_BACKEND IN_LIST orc_rt_log_backends)
  message(FATAL_ERROR "ORC_RT_LOG_BACKEND must be one of: ${orc_rt_log_backends}")
endif()
if(ORC_RT_LOG_BACKEND STREQUAL "os_log" AND NOT APPLE)
  message(FATAL_ERROR "ORC_RT_LOG_BACKEND=os_log is only supported on Apple platforms.")
endif()
string(TOUPPER "ORC_RT_LOG_BACKEND_${ORC_RT_LOG_BACKEND}" ORC_RT_LOG_BACKEND_VALUE)

# The "none" backend compiles all logging out, so the level has no effect. Warn
# if the user picked a non-default level under it, as their choice is ignored.
if(ORC_RT_LOG_BACKEND STREQUAL "none" AND
   NOT ORC_RT_LOG_LEVEL STREQUAL orc_rt_log_level_default)
  message(WARNING
    "ORC_RT_LOG_LEVEL is set to '${ORC_RT_LOG_LEVEL}', but ORC_RT_LOG_BACKEND is "
    "'none', so no logging is compiled in and the level has no effect. Select a "
    "backend (printf or os_log) to enable logging.")
endif()

# --- Generated config header ---
# Keep this last in the section so it captures every value derived above.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/orc-rt-c/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/orc-rt-c/config.h
  @ONLY
)

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
