##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a Next32 machine if available.
#
##===----------------------------------------------------------------------===##

set(NEXT32_LIB_NAME "next32")
set(NEXT32_MACHINE_TRIPLE "next32-unknown-linux-gnu")
set(NEXT32_PLUGIN_TARGET "omptarget.rtl.${NEXT32_LIB_NAME}")

################################################################################

set(LIBOMPTARGET_BUILD_NEXT32_PLUGIN TRUE CACHE BOOL
  "Whether to build ${NEXT32_MACHINE_NAME} plugin")
if (NOT LIBOMPTARGET_BUILD_NEXT32_PLUGIN)
  message(STATUS "Not building ${NEXT32_MACHINE_NAME} NextGen offloading plugin: LIBOMPTARGET_BUILD_NEXT32_PLUGIN is false")
  return()
endif()

################################################################################

message(STATUS "Building ${NEXT32_LIB_NAME} NextGen offloading plugin.")

add_target_library("${NEXT32_PLUGIN_TARGET}" ${NEXT32_LIB_NAME})

# Parent cmake file forbids having undefined symbols in the plugin, but at
# this point in the nextsilicon build process we don't yet have libnsapi.so,
# which contains the symbols needed for offloading to device. Remove the
# offending linker flag here.
get_target_property(PLUGIN_LINK_LIBS "${NEXT32_PLUGIN_TARGET}" LINK_LIBRARIES)
list(REMOVE_ITEM PLUGIN_LINK_LIBS "-Wl,-z,defs")
set_target_properties("${NEXT32_PLUGIN_TARGET}" PROPERTIES
  LINK_LIBRARIES "${PLUGIN_LINK_LIBS}")

target_sources(${NEXT32_PLUGIN_TARGET} PRIVATE src/rtl.cpp)

target_include_directories(omptarget.rtl.host PRIVATE
                           ${LIBOMPTARGET_INCLUDE_DIR})

target_compile_definitions(${NEXT32_PLUGIN_TARGET} PRIVATE
  TARGET_ELF_ID=EM_NEXT32)
target_compile_definitions(${NEXT32_PLUGIN_TARGET} PRIVATE
  LIBOMPTARGET_NEXTGEN_GENERIC_PLUGIN_TRIPLE="${NEXT32_LIB_NAME}")

list(APPEND LIBOMPTARGET_TESTED_PLUGINS "${NEXT32_PLUGIN_TARGET}")
set(LIBOMPTARGET_TESTED_PLUGINS
    "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
set(LIBOMPTARGET_SYSTEM_TARGETS 
    "${LIBOMPTARGET_SYSTEM_TARGETS} ${NEXT32_MACHINE_TRIPLE}" PARENT_SCOPE)
