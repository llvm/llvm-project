#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

# Unit test for cmake/Modules/GetTripleCMakeSystemName.cmake. Run via:
#   cmake -P get-triple-cmake-system-name.cmake
#
# _get_triple_cmake_system_name_raw must classify the same OS that
# Triple::normalize picks, including for the unnormalized/legacy triple
# shapes the runtimes build may feed in. The final block guards against the
# OS/environment tables drifting out of sync with TripleName.def.

cmake_minimum_required(VERSION 3.20)

# This script lives in llvm/test/tools/TargetParser/; the module is under the
# top-level cmake/, and TripleName.def under llvm/include/.
get_filename_component(_llvm_dir "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
get_filename_component(_project_dir "${_llvm_dir}/.." ABSOLUTE)

include("${_project_dir}/cmake/Modules/GetTripleCMakeSystemName.cmake")

# Accumulate failures in a global property so it survives function scopes; the
# final OK marker is only printed when it stays zero. This matters because the
# lit test pipes into FileCheck, which would otherwise miss a non-zero exit.
set_property(GLOBAL PROPERTY _gtcsn_failures 0)

function(_fail msg)
  message(SEND_ERROR "${msg}")
  get_property(_n GLOBAL PROPERTY _gtcsn_failures)
  math(EXPR _n "${_n} + 1")
  set_property(GLOBAL PROPERTY _gtcsn_failures "${_n}")
endfunction()

# Assert the classifier maps triple to expected.
function(expect triple expected)
  _get_triple_cmake_system_name_raw("${triple}" got)
  if(NOT got STREQUAL expected)
    _fail("'${triple}': got '${got}', expected '${expected}'")
  endif()
endfunction()

# Canonical triples, one per interesting mapping.
expect("x86_64-unknown-linux-gnu" "Linux")
expect("arm64-apple-darwin" "Darwin")
expect("s390x-ibm-zos" "OS390")
expect("aarch64-unknown-linux-android" "Android")
expect("x86_64-pc-windows-cygnus" "CYGWIN")
expect("wasm32-unknown-wasip1" "WASI")
expect("arm64-apple-ios" "iOS")
expect("riscv64-unknown-elf" "Generic")

# TRIPLE_OS_ALIAS rows resolve to their target OS's CMake name.
expect("x86_64-apple-macos" "Darwin")   # -> MacOSX
expect("x86_64-pc-win32" "Windows")     # -> Win32
expect("arm64-apple-visionos" "visionOS") # -> XROS

# Vendor omitted / OS or env out of the fixed position, version suffixes, and
# GCC-legacy spellings.
expect("aarch64-linux-android21" "Android")
expect("aarch64-unknown-linux-android21" "Android")
expect("arm-linux-androideabi" "Android")
expect("armv7a-linux-androideabi29" "Android")
expect("x86_64-linux-gnu" "Linux")
expect("riscv64-linux-gnu" "Linux")
expect("x86_64-apple-macosx10.15" "Darwin")
expect("armv7-apple-ios13.0" "iOS")
expect("x86_64-pc-freebsd14" "FreeBSD")

# The "real-world funky triples" from TripleTest.cpp's Normalization test.
expect("i386-mingw32" "Windows")   # -> i386-unknown-windows-gnu
expect("i486-linux-gnu" "Linux")   # -> i486-unknown-linux-gnu
expect("i386-redhat-linux" "Linux")
expect("i686-linux" "Linux")
expect("arm-none-eabi" "Generic")  # no OS
expect("ve-linux" "Linux")
expect("wasm32-wasi" "WASI")
expect("wasm64-wasi" "WASI")
expect("x86_64-pc-cygwin" "CYGWIN")   # -> windows-cygnus
expect("x86_64-pc-msys" "CYGWIN")     # -> windows-cygnus
expect("x86_64-w64-mingw32" "Windows")
expect("i686-w64-mingw32" "Windows")

# OS/environment special-cases that Triple::normalize applies.
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

# firmware has no dedicated CMake name and only maps to Darwin for the apple
# vendor; any other vendor is Generic.
expect("arm-none-firmware" "Generic")
expect("arm-unknown-firmware" "Generic")
expect("arm-pc-firmware" "Generic")

# driverkit and bridgeos are always Apple, so they map to Darwin regardless of
# the vendor component.
foreach(_os driverkit bridgeos)
  foreach(_vendor apple none unknown pc)
    expect("arm-${_vendor}-${_os}" "Darwin")
  endforeach()
endforeach()

# A lone OS component with no arch is a valid clang target (e.g.
# --target=darwin -> unknown-unknown-darwin), so it classifies by the first
# (and only) component.
expect("darwin" "Darwin")
expect("linux" "Linux")
expect("ios" "iOS")
expect("freebsd" "FreeBSD")
expect("wasi" "WASI")
expect("zos" "OS390")
expect("mingw32" "Windows")
expect("cygwin" "CYGWIN")

# Only when no component classifies as an OS does the classifier return empty
# (the caller then falls back to the host system name).
expect("x86_64" "")
expect("arm-none" "")

# Data-driven: a canonical arch-unknown-<os> triple maps to the CMake name the
# module declares for every OS/alias row.
_gtcsn_populate_tables()
list(LENGTH _GTCSN_OS_NAMES _nos)
math(EXPR _last_os "${_nos} - 1")
foreach(_i RANGE ${_last_os})
  list(GET _GTCSN_OS_NAMES ${_i} _name)
  list(GET _GTCSN_OS_CMAKE ${_i} _cmake)
  expect("arch-unknown-${_name}" "${_cmake}")
endforeach()

# A name that is a prefix of a LATER name would shadow it under prefix matching,
# so within each table a name must never be a prefix of an earlier one.
function(check_prefix_order)
  foreach(_list _GTCSN_OS_NAMES _GTCSN_ENV_NAMES)
    set(_names "${${_list}}")
    list(LENGTH _names _n)
    math(EXPR _last "${_n} - 1")
    foreach(_i RANGE ${_last})
      list(GET _names ${_i} _short)
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

# Drift guard: every OS/alias name in TripleName.def must have a mapping entry
# in the module, and every environment override the module relies on must still
# name a real environment in the .def. The CMake system names themselves live
# only in the module (they are a CMake property, not part of llvm::Triple), so
# the .def is only cross-checked for the set of recognized names.
set(_def "${_llvm_dir}/include/llvm/TargetParser/TripleName.def")
if(NOT EXISTS "${_def}")
  message(FATAL_ERROR "TripleName.def not found at ${_def}")
endif()

file(STRINGS "${_def}" _os_lines REGEX "^[ \t]*TRIPLE_OS(_ALIAS)?\\(")
foreach(_line IN LISTS _os_lines)
  if(_line MATCHES "\"([^\"]+)\"")
    set(_name "${CMAKE_MATCH_1}")
    if(NOT _name IN_LIST _GTCSN_OS_NAMES)
      _fail("TripleName.def OS '${_name}' has no CMake system-name mapping "
            "in GetTripleCMakeSystemName.cmake")
    endif()
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
