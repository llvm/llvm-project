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

if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building AMDGPU NextGen plugin: only support AMDGPU in Linux x86_64, ppc64le, or aarch64 hosts")
  return()
endif()

################################################################################
# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=AMDGPU)

# Define debug prefix. TODO: This should be automatized in the Debug.h but it
# requires changing the original plugins.
add_definitions(-DDEBUG_PREFIX="TARGET AMDGPU RTL")

set(LIBOMPTARGET_DLOPEN_LIBHSA OFF)
option(LIBOMPTARGET_FORCE_DLOPEN_LIBHSA "Build with dlopened libhsa" ${LIBOMPTARGET_DLOPEN_LIBHSA})

if (${hsa-runtime64_FOUND} AND NOT LIBOMPTARGET_FORCE_DLOPEN_LIBHSA)
  libomptarget_say("Building AMDGPU NextGen plugin linked against libhsa")
  set(LIBOMPTARGET_EXTRA_SOURCE)
  set(LIBOMPTARGET_DEP_LIBRARIES hsa-runtime64::hsa-runtime64)
else()
  libomptarget_say("Building AMDGPU NextGen plugin for dlopened libhsa")
  include_directories(dynamic_hsa)
  set(LIBOMPTARGET_EXTRA_SOURCE dynamic_hsa/hsa.cpp)
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

add_llvm_library(omptarget.rtl.amdgpu SHARED
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
  PluginCommon
  ${LIBOMPTARGET_DEP_LIBRARIES}
  ${OPENMP_PTHREAD_LIB}
  ${LDFLAGS_UNDEFINED}

  NO_INSTALL_RPATH
)

if ((OMPT_TARGET_DEFAULT) AND (LIBOMPTARGET_OMPT_SUPPORT))
  target_link_libraries(omptarget.rtl.amdgpu PRIVATE OMPT)
endif()

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(omptarget.rtl.amdgpu PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")
endif()

target_include_directories(
  omptarget.rtl.amdgpu
  PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Configure testing for the AMDGPU plugin. We will build tests if we could a
# functional AMD GPU on the system, or if manually specifies by the user.
option(LIBOMPTARGET_FORCE_AMDGPU_TESTS "Build AMDGPU libomptarget tests" OFF)
if (LIBOMPTARGET_FOUND_AMDGPU_GPU OR LIBOMPTARGET_FORCE_AMDGPU_TESTS)
  # Report to the parent scope that we are building a plugin for amdgpu
  set(LIBOMPTARGET_SYSTEM_TARGETS 
      "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn-amd-amdhsa" PARENT_SCOPE)
  list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.amdgpu")
  set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
else()
  libomptarget_say("Not generating AMDGPU tests, no supported devices detected."
                   " Use 'LIBOMPTARGET_FORCE_AMDGPU_TESTS' to override.")
  return()
endif()

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.amdgpu LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
set_target_properties(omptarget.rtl.amdgpu PROPERTIES
  INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_VISIBILITY_PRESET protected)
