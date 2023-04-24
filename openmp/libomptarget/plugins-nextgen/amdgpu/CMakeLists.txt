##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for an AMDGPU machine if available.
#
##===----------------------------------------------------------------------===##

################################################################################
set(LIBOMPTARGET_BUILD_AMDGPU_PLUGIN TRUE CACHE BOOL
  "Whether to build AMDGPU plugin")
if (NOT LIBOMPTARGET_BUILD_AMDGPU_PLUGIN)
  libomptarget_say("Not building AMDGPU NextGen offloading plugin: LIBOMPTARGET_BUILD_AMDGPU_PLUGIN is false")
  return()
endif()

# as of rocm-3.7, hsa is installed with cmake packages and kmt is found via hsa
find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  libomptarget_say("Not building AMDGPU NextGen plugin: only support AMDGPU in Linux x86_64, ppc64le, or aarch64 hosts")
  return()
endif()

################################################################################
# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=AMDGPU)

# Define debug prefix. TODO: This should be automatized in the Debug.h but it
# requires changing the original plugins.
add_definitions(-DDEBUG_PREFIX="TARGET AMDGPU RTL")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc64le)|(aarch64)$")
   add_definitions(-DLITTLEENDIAN_CPU=1)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DDEBUG)
endif()

set(LIBOMPTARGET_DLOPEN_LIBHSA OFF)
option(LIBOMPTARGET_FORCE_DLOPEN_LIBHSA "Build with dlopened libhsa" ${LIBOMPTARGET_DLOPEN_LIBHSA})

if (${hsa-runtime64_FOUND} AND NOT LIBOMPTARGET_FORCE_DLOPEN_LIBHSA)
  libomptarget_say("Building AMDGPU NextGen plugin linked against libhsa")
  set(LIBOMPTARGET_EXTRA_SOURCE)
  set(LIBOMPTARGET_DEP_LIBRARIES hsa-runtime64::hsa-runtime64)
else()
  libomptarget_say("Building AMDGPU NextGen plugin for dlopened libhsa")
  include_directories(../../plugins/amdgpu/dynamic_hsa)
  set(LIBOMPTARGET_EXTRA_SOURCE ../../plugins/amdgpu/dynamic_hsa/hsa.cpp)
  set(LIBOMPTARGET_DEP_LIBRARIES)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  # On FreeBSD, the 'environ' symbol is undefined at link time, but resolved by
  # the dynamic linker at runtime. Therefore, allow the symbol to be undefined
  # when creating a shared library.
  set(LDFLAGS_UNDEFINED "-Wl,--allow-shlib-undefined")
else()
  set(LDFLAGS_UNDEFINED "-Wl,-z,defs")
endif()

add_llvm_library(omptarget.rtl.amdgpu.nextgen SHARED
  src/rtl.cpp
  ${LIBOMPTARGET_EXTRA_SOURCE}

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/utils

  LINK_COMPONENTS
  Support
  Object

  LINK_LIBS
  PRIVATE
  elf_common
  MemoryManager
  PluginInterface
  ${LIBOMPTARGET_DEP_LIBRARIES}
  ${OPENMP_PTHREAD_LIB}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
  ${LDFLAGS_UNDEFINED}

  NO_INSTALL_RPATH
)

target_include_directories(
  omptarget.rtl.amdgpu.nextgen
  PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
)


# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.amdgpu.nextgen LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
set_target_properties(omptarget.rtl.amdgpu.nextgen PROPERTIES
  INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_VISIBILITY_PRESET protected)
