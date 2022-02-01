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
  libomptarget_say("Not building AMDGPU offloading plugin: LIBOMPTARGET_BUILD_AMDGPU_PLUGIN is false")
  return()
endif()

# as of rocm-3.7, hsa is installed with cmake packages and kmt is found via hsa
find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)

if(NOT LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building AMDGPU plugin: LIBELF not found")
  return()
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  libomptarget_say("Not building AMDGPU plugin: only support AMDGPU in Linux x86_64, ppc64le, or aarch64 hosts")
  return()
endif()

if (NOT LIBOMPTARGET_LLVM_INCLUDE_DIRS)
  libomptarget_say("Not building AMDGPU plugin: Missing definition for LIBOMPTARGET_LLVM_INCLUDE_DIRS")
  return()
endif()

################################################################################
# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=AMDGPU)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc64le)|(aarch64)$")
   add_definitions(-DLITTLEENDIAN_CPU=1)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_definitions(-DDEBUG)
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
)

set(LIBOMPTARGET_DLOPEN_LIBHSA OFF)
option(LIBOMPTARGET_FORCE_DLOPEN_LIBHSA "Build with dlopened libhsa" ${LIBOMPTARGET_DLOPEN_LIBHSA})

if (${hsa-runtime64_FOUND} AND NOT LIBOMPTARGET_FORCE_DLOPEN_LIBHSA)
  libomptarget_say("Building AMDGPU plugin linked against libhsa")
  set(LIBOMPTARGET_EXTRA_SOURCE)
  set(LIBOMPTARGET_DEP_LIBRARIES hsa-runtime64::hsa-runtime64)
else()
  libomptarget_say("Building AMDGPU plugin for dlopened libhsa")
  include_directories(dynamic_hsa)
  set(LIBOMPTARGET_EXTRA_SOURCE dynamic_hsa/hsa.cpp)
  set(LIBOMPTARGET_DEP_LIBRARIES)
endif()

add_library(omptarget.rtl.amdgpu SHARED
      impl/impl.cpp
      impl/interop_hsa.cpp
      impl/data.cpp
      impl/get_elf_mach_gfx_name.cpp
      impl/system.cpp
      impl/msgpack.cpp
      src/rtl.cpp
      ${LIBOMPTARGET_EXTRA_SOURCE}
      )

# Install plugin under the lib destination folder.
# When we build for debug, OPENMP_LIBDIR_SUFFIX get set to -debug
install(TARGETS omptarget.rtl.amdgpu LIBRARY DESTINATION "lib${OPENMP_LIBDIR_SUFFIX}")
set_property(TARGET omptarget.rtl.amdgpu PROPERTY INSTALL_RPATH_USE_LINK_PATH ON)

if(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
  # On FreeBSD, the 'environ' symbol is undefined at link time, but resolved by
  # the dynamic linker at runtime. Therefore, allow the symbol to be undefined
  # when creating a shared library.
  set(LDFLAGS_UNDEFINED "-Wl,--allow-shlib-undefined")
else()
  set(LDFLAGS_UNDEFINED "-Wl,-z,defs")
endif()

target_link_libraries(
  omptarget.rtl.amdgpu
  PRIVATE
  elf_common
  ${LIBOMPTARGET_DEP_LIBRARIES}
  ${CMAKE_DL_LIBS}
  ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
  ${OPENMP_PTHREAD_LIB}
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
  ${LDFLAGS_UNDEFINED}
  )

# in case of amdgcn, skip running tests if amdgpu-arch was not built or fails
if (NOT TARGET amdgpu-arch)
  libomptarget_say("Not generating amdgcn test targets as amdgpu-arch is not found")
  return()
endif()

get_property(AMDGPU_ARCH_COMMAND TARGET amdgpu-arch PROPERTY LOCATION)

execute_process(COMMAND ${AMDGPU_ARCH_COMMAND} RESULT_VARIABLE amdgpu_arch_result
                                               OUTPUT_VARIABLE amdgpu_arch_output)
if (${amdgpu_arch_result})
  libomptarget_say("Not generating amdgcn test targets as amdgpu-arch exited with ${amdgpu_arch_result}")
else()
  # Report to the parent scope that we are building a plugin for amdgpu
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn-amd-amdhsa-newRTL " PARENT_SCOPE)
endif()

