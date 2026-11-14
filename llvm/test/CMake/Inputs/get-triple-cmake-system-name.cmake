#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

# Test driver for cmake/Modules/GetTripleCMakeSystemName.cmake.
#
# Usage:
#   cmake -DSRC_ROOT=<llvm source root> -P get-triple-cmake-system-name.cmake

cmake_minimum_required(VERSION 3.20)

if(NOT SRC_ROOT)
  message(FATAL_ERROR "SRC_ROOT is not set")
endif()

get_filename_component(_llvm_dir "${SRC_ROOT}" ABSOLUTE)

include("${_llvm_dir}/../cmake/Modules/GetTripleCMakeSystemName.cmake")

set_property(GLOBAL PROPERTY _gtcsn_failures 0)

function(_fail)
  string(JOIN "" _msg ${ARGN})
  message(SEND_ERROR "${_msg}")
  get_property(_n GLOBAL PROPERTY _gtcsn_failures)
  math(EXPR _n "${_n} + 1")
  set_property(GLOBAL PROPERTY _gtcsn_failures "${_n}")
endfunction()

function(expect triple expected)
  _get_triple_cmake_system_name_raw("${triple}" got)
  if(NOT got STREQUAL expected)
    _fail("'${triple}': got '${got}', expected '${expected}'")
  endif()
endfunction()

expect("x86_64-unknown-linux-gnu" "Linux")
expect("arm64-apple-darwin" "Darwin")
expect("s390x-ibm-zos" "OS390")
expect("aarch64-unknown-linux-android" "Android")
expect("x86_64-pc-windows-cygnus" "CYGWIN")
expect("wasm32-unknown-wasip1" "WASI")
expect("arm64-apple-ios" "iOS")
expect("riscv64-unknown-elf" "Generic")

# OS aliases
expect("x86_64-apple-macos" "Darwin")
expect("x86_64-pc-win32" "Windows")
expect("arm64-apple-visionos" "visionOS")

# Missing vendor and version suffixes
expect("aarch64-linux-android21" "Android")
expect("aarch64-unknown-linux-android21" "Android")
expect("arm-linux-androideabi" "Android")
expect("armv7a-linux-androideabi29" "Android")
expect("x86_64-linux-gnu" "Linux")
expect("riscv64-linux-gnu" "Linux")
expect("x86_64-apple-macosx10.15" "Darwin")
expect("armv7-apple-ios13.0" "iOS")
expect("x86_64-pc-freebsd14" "FreeBSD")

# From the Normalization test in TripleTest.cpp
expect("i386-mingw32" "Windows")
expect("i486-linux-gnu" "Linux")
expect("i386-redhat-linux" "Linux")
expect("i686-linux" "Linux")
expect("arm-none-eabi" "Generic")
expect("ve-linux" "Linux")
expect("wasm32-wasi" "WASI")
expect("wasm64-wasi" "WASI")
expect("x86_64-pc-cygwin" "CYGWIN")
expect("x86_64-pc-msys" "CYGWIN")
expect("x86_64-w64-mingw32" "Windows")
expect("i686-w64-mingw32" "Windows")

expect("x86_64-pc-windows" "Windows")
expect("x86_64-pc-windows-msvc" "Windows")
expect("x86_64-pc-windows-gnu" "Windows")
expect("x86_64-apple-macosx" "Darwin")
expect("arm-apple-darwin" "Darwin")
expect("arm-apple-ios" "iOS")
expect("arm-apple-tvos" "tvOS")
expect("arm-apple-watchos" "watchOS")
expect("arm-apple-xros" "visionOS")
expect("arm-apple-driverkit" "Darwin")
expect("arm-apple-bridgeos" "Darwin")
expect("arm-apple-firmware" "Darwin")

# firmware is only Darwin with the apple vendor.
expect("arm-none-firmware" "Generic")
expect("arm-unknown-firmware" "Generic")
expect("arm-pc-firmware" "Generic")

foreach(_os driverkit bridgeos)
  foreach(_vendor apple none unknown pc)
    expect("arm-${_vendor}-${_os}" "Darwin")
  endforeach()
endforeach()

# Bare OS, e.g. --target=darwin
expect("darwin" "Darwin")
expect("linux" "Linux")
expect("ios" "iOS")
expect("freebsd" "FreeBSD")
expect("wasi" "WASI")
expect("zos" "OS390")
expect("mingw32" "Windows")
expect("cygwin" "CYGWIN")

# No OS
expect("x86_64" "")
expect("arm-none" "")

function(expect_public triple expected)
  get_triple_cmake_system_name("${triple}" got)
  if(NOT got STREQUAL expected)
    _fail("get_triple_cmake_system_name('${triple}'): got '${got}', "
          "expected '${expected}'")
  endif()
endfunction()

set(CMAKE_HOST_SYSTEM_NAME "HostOS")
expect_public("x86_64" "HostOS")
expect_public("x86_64-unknown-linux-gnu" "Linux")

# Every table entry
_gtcsn_populate_tables()
list(LENGTH _GTCSN_OS_NAMES _nos)
math(EXPR _last_os "${_nos} - 1")
foreach(_i RANGE ${_last_os})
  list(GET _GTCSN_OS_NAMES ${_i} _name)
  list(GET _GTCSN_OS_CMAKE ${_i} _cmake)
  expect("arch-unknown-${_name}" "${_cmake}")
endforeach()

# Names are used as regexes, and an earlier name must not be a prefix of a later
# one.
function(check_prefix_order)
  foreach(_list _GTCSN_OS_NAMES _GTCSN_ENV_NAMES)
    set(_names "${${_list}}")
    list(LENGTH _names _n)
    math(EXPR _last "${_n} - 1")
    foreach(_i RANGE ${_last})
      list(GET _names ${_i} _short)
      if(NOT _short MATCHES "^[a-z0-9]+$")
        _fail("${_list}: '${_short}' is not lowercase alphanumeric")
      endif()

      math(EXPR _next "${_i} + 1")
      if(_next GREATER _last)
        continue()
      endif()

      foreach(_j RANGE ${_next} ${_last})
        list(GET _names ${_j} _longer)
        string(FIND "${_longer}" "${_short}" _pos)
        if(_pos EQUAL 0 AND NOT _longer STREQUAL _short)
          _fail("${_list}: '${_short}' precedes and shadows '${_longer}'")
        endif()
      endforeach()
    endforeach()
  endforeach()
endfunction()
check_prefix_order()

# The OS names must match TripleName.def, and the environment names must exist
# there.
set(_def "${_llvm_dir}/include/llvm/TargetParser/TripleName.def")
if(NOT EXISTS "${_def}")
  message(FATAL_ERROR "TripleName.def not found at ${_def}")
endif()

file(STRINGS "${_def}" _os_lines REGEX "^[ \t]*TRIPLE_OS(_ALIAS)?\\(")
set(_def_os_names "")
foreach(_line IN LISTS _os_lines)
  if(_line MATCHES "\"([^\"]+)\"")
    list(APPEND _def_os_names "${CMAKE_MATCH_1}")
  endif()
endforeach()

foreach(_name IN LISTS _def_os_names)
  if(NOT _name IN_LIST _GTCSN_OS_NAMES)
    _fail("TripleName.def OS '${_name}' has no CMake system-name mapping "
          "in GetTripleCMakeSystemName.cmake")
  endif()
endforeach()

foreach(_name IN LISTS _GTCSN_OS_NAMES)
  if(NOT _name IN_LIST _def_os_names)
    _fail("GetTripleCMakeSystemName.cmake OS '${_name}' is no longer an OS "
          "in TripleName.def")
  endif()
endforeach()

file(STRINGS "${_def}" _env_lines REGEX "^[ \t]*TRIPLE_ENV\\(")
set(_def_env_names "")

foreach(_line IN LISTS _env_lines)
  if(_line MATCHES "\"([^\"]+)\"")
    list(APPEND _def_env_names "${CMAKE_MATCH_1}")
  endif()
endforeach()

foreach(_name IN LISTS _GTCSN_ENV_NAMES)
  if(NOT _name IN_LIST _def_env_names)
    _fail("GetTripleCMakeSystemName.cmake environment override '${_name}' "
          "is no longer an environment in TripleName.def")
  endif()
endforeach()

get_property(_failures GLOBAL PROPERTY _gtcsn_failures)
if(_failures GREATER 0)
  message(FATAL_ERROR "${_failures} case(s) failed")
endif()

message(STATUS "get-triple-cmake-system-name: OK")
