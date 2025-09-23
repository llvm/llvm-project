# Finds the Microsoft DIA SDK and sets DIASDK_FOUND and related variables.
#
# This module is intended to be used both internally by LLVM's build system and
# by consuming projects when loading LLVMConfig.cmake.
#
# LLVM_WINSYSROOT may be set for locating the DIA SDK.
#
# If successful, the following variables will be defined:
#   DIASDK_FOUND
#   DIASDK_INCLUDE_DIR
#   DIASDK_LIBRARIES
#
# Additionally, the following import target will be defined:
#   DIASDK::Diaguids

if(NOT WIN32)
  set(DIASDK_FOUND FALSE)
  return()
endif()

if(LLVM_WINSYSROOT)
  set(MSVC_DIA_SDK_DIR "${LLVM_WINSYSROOT}/DIA SDK" CACHE PATH
      "Path to the DIA SDK")
else()
  set(MSVC_DIA_SDK_DIR "$ENV{VSINSTALLDIR}DIA SDK" CACHE PATH
      "Path to the DIA SDK")
endif()

find_path(DIASDK_INCLUDE_DIR
  NAMES dia2.h
  PATHS "${MSVC_DIA_SDK_DIR}/include"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

if(IS_DIRECTORY "${MSVC_DIA_SDK_DIR}")
  set(_DIA_SDK_LIB_DIR "${MSVC_DIA_SDK_DIR}/lib")

  if("$ENV{VSCMD_ARG_TGT_ARCH}" STREQUAL "arm64")
    set(_DIA_SDK_LIB_DIR "${_DIA_SDK_LIB_DIR}/arm64")
  elseif("$ENV{VSCMD_ARG_TGT_ARCH}" STREQUAL "arm")
    set(_DIA_SDK_LIB_DIR "${_DIA_SDK_LIB_DIR}/arm")
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_DIA_SDK_LIB_DIR "${_DIA_SDK_LIB_DIR}/amd64")
  endif()

  find_library(DIASDK_LIBRARIES
    NAMES diaguids
    PATHS "${_DIA_SDK_LIB_DIR}"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  DIASDK
  FOUND_VAR
    DIASDK_FOUND
  REQUIRED_VARS
    DIASDK_INCLUDE_DIR
    DIASDK_LIBRARIES
)
mark_as_advanced(DIASDK_INCLUDE_DIR DIASDK_LIBRARIES)

if(DIASDK_FOUND)
  if(NOT TARGET DIASDK::Diaguids)
    add_library(DIASDK::Diaguids UNKNOWN IMPORTED)
    set_target_properties(DIASDK::Diaguids PROPERTIES
      IMPORTED_LOCATION "${DIASDK_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${DIASDK_INCLUDE_DIR}"
    )
  endif()
endif()
