cmake_minimum_required(VERSION 3.20)
project(WindowsVersionResource NONE)
list(APPEND CMAKE_MODULE_PATH
  "${LLVM_SRC_ROOT}/cmake/modules"
  "${LLVM_SRC_ROOT}/../cmake/Modules")
include(AddLLVM)

function(check_version label)
  set(resource_file "${CMAKE_CURRENT_BINARY_DIR}/${label}.rc")
  set_windows_version_resource_properties(probe "${resource_file}" ${ARGN})
  get_source_file_property(definitions "${resource_file}" COMPILE_DEFINITIONS)
  list(FILTER definitions INCLUDE REGEX "^RC_(FILE|PRODUCT)_VERSION=")
  message("${label}: ${definitions}")
endfunction()

foreach(version IN ITEMS "24.0.0git" "24.0.0" "Vendor 24.0.0")
  set(PACKAGE_VERSION "${version}")
  check_version("${version}")
endforeach()
check_version(override VERSION_STRING "23.0.0rc1")
unset(PACKAGE_VERSION)
check_version(unset)
set(PACKAGE_VERSION "")
check_version(empty)
