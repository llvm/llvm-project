##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a Next32 machine if available.
#
##===----------------------------------------------------------------------===##

set(NEXT32_MACHINE_NAME "next32")
set(NEXT32_LIB_NAME "next32")
set(NEXT32_MACHINE_TRIPLE "next32-unknown-linux-gnu")

################################################################################

set(LIBOMPTARGET_BUILD_NEXT32_PLUGIN TRUE CACHE BOOL
  "Whether to build ${NEXT32_MACHINE_NAME} plugin")
if (NOT LIBOMPTARGET_BUILD_NEXT32_PLUGIN)
  libomptarget_say("Not building ${NEXT32_MACHINE_NAME} NextGen offloading plugin: LIBOMPTARGET_BUILD_NEXT32_PLUGIN is false")
  return()
endif()

################################################################################

libomptarget_say("Building ${NEXT32_MACHINE_NAME} NextGen offloading plugin.")

# Define macro to be used as prefix of the runtime messages for this target.
add_definitions("-DTARGET_NAME=${NEXT32_MACHINE_NAME}")

# Define debug prefix. TODO: This should be automatized in the Debug.h but
# it requires changing the original plugins.
add_definitions(-DDEBUG_PREFIX="TARGET ${NEXT32_MACHINE_NAME} RTL")

# Define macro with the ELF ID for this target.
add_definitions("-DTARGET_ELF_ID=0xDA7A")

# Define target regiple
add_definitions("-DLIBOMPTARGET_NEXTGEN_GENERIC_PLUGIN_TRIPLE=${NEXT32_MACHINE_NAME}")

add_llvm_library("omptarget.rtl.${NEXT32_LIB_NAME}"
  SHARED

  ${CMAKE_CURRENT_SOURCE_DIR}/src/rtl.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}

    LINK_LIBS
    PRIVATE
    OMPT
    PluginCommon
    ${OPENMP_PTHREAD_LIB}

  NO_INSTALL_RPATH
)

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries("omptarget.rtl.${NEXT32_LIB_NAME}" PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")
endif()

# Install plugin under the lib destination folder.
install(TARGETS "omptarget.rtl.${NEXT32_LIB_NAME}"
  LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
set_target_properties("omptarget.rtl.${NEXT32_LIB_NAME}" PROPERTIES
  INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_VISIBILITY_PRESET protected)

target_include_directories("omptarget.rtl.${NEXT32_LIB_NAME}" PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR})

list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.${NEXT32_LIB_NAME}")
set(LIBOMPTARGET_TESTED_PLUGINS
    "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
set(LIBOMPTARGET_SYSTEM_TARGETS
    "${LIBOMPTARGET_SYSTEM_TARGETS} ${NEXT32_MACHINE_TRIPLE}" PARENT_SCOPE)
