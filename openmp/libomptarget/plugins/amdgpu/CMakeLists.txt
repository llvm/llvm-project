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

if(NOT LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building AMDGPU plugin: LIBELF not found")
  return()
endif()

if(NOT ROCM_DIR)
  libomptarget_say("Not building AMDGPU plugin: ROCM_DIR is not set")
  return()
endif()

set(LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS ${ROCM_DIR}/hsa/include ${ROCM_DIR}/hsa/include/hsa)
set(LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS ${ROCM_DIR}/hsa/lib)
set(LIBOMPTARGET_DEP_LIBHSAKMT_LIBRARIES_DIRS ${ROCM_DIR}/lib)

mark_as_advanced( LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  libomptarget_say("Not building amdgpu plugin: only support amdgpu in Linux x86_64, ppc64le, or aarch64 hosts.")
  return()
endif()
libomptarget_say("Building amdgpu offloading plugin using ROCM_DIR = ${ROCM_DIR}")

libomptarget_say("LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS: ${LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS}")
libomptarget_say("LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS ${LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS}")
libomptarget_say("LIBOMPTARGET_DEP_LIBHSAKMT_LIBRARIES_DIRS: ${LIBOMPTARGET_DEP_LIBHSAKMT_LIBRARIES_DIRS}")

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
  ${LIBOMPTARGET_DEP_LIBHSA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
)

add_library(omptarget.rtl.amdgpu SHARED
      impl/atmi.cpp
      impl/atmi_interop_hsa.cpp
      impl/data.cpp
      impl/machine.cpp
      impl/system.cpp
      impl/utils.cpp
      impl/msgpack.cpp
      src/rtl.cpp 
      )

# Install plugin under the lib destination folder.
# When we build for debug, OPENMP_LIBDIR_SUFFIX get set to -debug
install(TARGETS omptarget.rtl.amdgpu LIBRARY DESTINATION "lib${OPENMP_LIBDIR_SUFFIX}")

target_link_libraries(
  omptarget.rtl.amdgpu
  -lpthread -ldl -Wl,-rpath,${OPENMP_INSTALL_LIBDIR}
  -L${LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS} -L${LIBOMPTARGET_DEP_LIBHSAKMT_LIBRARIES_DIRS} -lhsa-runtime64 -lhsakmt -Wl,-rpath,${LIBOMPTARGET_DEP_LIBHSA_LIBRARIES_DIRS},-rpath,${LIBOMPTARGET_DEP_LIBHSAKMT_LIBRARIES_DIRS}
  -lelf
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
  "-Wl,-z,defs"
  )

# Report to the parent scope that we are building a plugin for amdgpu
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn-amd-amdhsa" PARENT_SCOPE)

