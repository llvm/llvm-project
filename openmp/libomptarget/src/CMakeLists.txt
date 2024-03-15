##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build offloading library libomptarget.so.
#
##===----------------------------------------------------------------------===##

libomptarget_say("Building offloading runtime library libomptarget.")

add_llvm_library(omptarget
  SHARED

  device.cpp
  interface.cpp
  omptarget.cpp
  OffloadRTL.cpp
  LegacyAPI.cpp
  PluginManager.cpp
  DeviceImage.cpp

  OpenMP/API.cpp
  OpenMP/Mapping.cpp
  OpenMP/InteropAPI.cpp
  OpenMP/OMPT/Callback.cpp


  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}

  LINK_COMPONENTS
  Support
  Object

  LINK_LIBS
  PUBLIC
  omp

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)
target_include_directories(omptarget PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(omptarget PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(omptarget PRIVATE
  TARGET_NAME=omptarget
  DEBUG_PREFIX="omptarget"
)

target_compile_options(omptarget PUBLIC ${offload_compile_flags})
target_link_options(omptarget PUBLIC ${offload_link_flags})

macro(check_plugin_target target)
if (TARGET omptarget.rtl.${target})
	list(APPEND LIBOMPTARGET_PLUGINS_TO_LOAD ${target})
endif()
endmacro()

set(LIBOMPTARGET_PLUGINS_TO_LOAD "" CACHE STRING
  "Comma separated list of plugin names to look for at runtime")
if (NOT LIBOMPTARGET_PLUGINS_TO_LOAD)
	check_plugin_target(ppc64)
	check_plugin_target(x86_64)
	check_plugin_target(cuda)
	check_plugin_target(aarch64)
	check_plugin_target(amdgpu)
endif()

list(TRANSFORM LIBOMPTARGET_PLUGINS_TO_LOAD PREPEND "\"libomptarget.rtl.")
list(TRANSFORM LIBOMPTARGET_PLUGINS_TO_LOAD APPEND "\"")
list(JOIN LIBOMPTARGET_PLUGINS_TO_LOAD "," ENABLED_OFFLOAD_PLUGINS)
target_compile_definitions(omptarget PRIVATE ENABLED_OFFLOAD_PLUGINS=${ENABLED_OFFLOAD_PLUGINS})

# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(omptarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS omptarget LIBRARY COMPONENT omptarget DESTINATION "${OPENMP_INSTALL_LIBDIR}")
