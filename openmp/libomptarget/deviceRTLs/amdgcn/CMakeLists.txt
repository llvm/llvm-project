##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build the AMDGCN Device RTL if the ROCM tools are available
#
##===----------------------------------------------------------------------===##

find_package(LLVM QUIET CONFIG
  PATHS
  $ENV{AOMP}
  $ENV{HOME}/rocm/aomp
  /opt/rocm/aomp
  /usr/lib/rocm/aomp
  ${LIBOMPTARGET_NVPTX_CUDA_COMPILER_DIR}
  ${LIBOMPTARGET_NVPTX_CUDA_LINKER_DIR}
  ${CMAKE_CXX_COMPILER_DIR}
  NO_DEFAULT_PATH)

if (LLVM_DIR)
  libomptarget_say("Found LLVM ${LLVM_PACKAGE_VERSION}. Configure: ${LLVM_DIR}/LLVMConfig.cmake")
else()
  libomptarget_say("Not building AMDGCN device RTL: AOMP not found")
  return()
endif()

set(AOMP_INSTALL_PREFIX ${LLVM_INSTALL_PREFIX})

if (AOMP_INSTALL_PREFIX)
  set(AOMP_BINDIR ${AOMP_INSTALL_PREFIX}/bin)
else()
  set(AOMP_BINDIR ${LLVM_BUILD_BINARY_DIR}/bin)
endif()

libomptarget_say("Building AMDGCN device RTL. LLVM_COMPILER_PATH=${AOMP_BINDIR}")

project(omptarget-amdgcn)

add_custom_target(omptarget-amdgcn ALL)

#optimization level
set(optimization_level 2)

# Activate RTL message dumps if requested by the user.
if(LIBOMPTARGET_NVPTX_DEBUG)
  set(CUDA_DEBUG -DOMPTARGET_NVPTX_DEBUG=-1)
endif()

get_filename_component(devicertl_base_directory
  ${CMAKE_CURRENT_SOURCE_DIR}
  DIRECTORY)

set(cuda_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/amdgcn_smid.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/src/amdgcn_locks.hip
  ${CMAKE_CURRENT_SOURCE_DIR}/src/target_impl.hip
  ${devicertl_base_directory}/common/src/cancel.cu
  ${devicertl_base_directory}/common/src/critical.cu
  ${devicertl_base_directory}/common/src/data_sharing.cu
  ${devicertl_base_directory}/common/src/libcall.cu
  ${devicertl_base_directory}/common/src/loop.cu
  ${devicertl_base_directory}/common/src/omp_data.cu
  ${devicertl_base_directory}/common/src/omptarget.cu
  ${devicertl_base_directory}/common/src/parallel.cu
  ${devicertl_base_directory}/common/src/reduction.cu
  ${devicertl_base_directory}/common/src/support.cu
  ${devicertl_base_directory}/common/src/sync.cu
  ${devicertl_base_directory}/common/src/task.cu)

set(h_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/amdgcn_interface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/hip_atomics.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/target_impl.h
  ${devicertl_base_directory}/common/debug.h
  ${devicertl_base_directory}/common/device_environment.h
  ${devicertl_base_directory}/common/omptarget.h
  ${devicertl_base_directory}/common/omptargeti.h
  ${devicertl_base_directory}/common/state-queue.h
  ${devicertl_base_directory}/common/target_atomic.h
  ${devicertl_base_directory}/common/state-queuei.h
  ${devicertl_base_directory}/common/support.h)

# for both in-tree and out-of-tree build
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
else()
  set(OUTPUTDIR ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endif()

# create libraries
set(mcpus gfx700 gfx701 gfx801 gfx803 gfx900)
if (DEFINED LIBOMPTARGET_AMDGCN_GFXLIST)
  set(mcpus ${LIBOMPTARGET_AMDGCN_GFXLIST})
endif()

macro(add_cuda_bc_library)
  set(cu_cmd ${AOMP_BINDIR}/clang++
    -std=c++14
    -fcuda-rdc
    -fvisibility=default
    --cuda-device-only
    -Wno-unused-value
    -x hip
    -nogpulib -nogpuinc
    -O${optimization_level}
    --cuda-gpu-arch=${mcpu}
    ${CUDA_DEBUG}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${devicertl_base_directory})

  set(bc1_files)

  foreach(file ${ARGN})
    get_filename_component(fname ${file} NAME_WE)
    set(bc1_filename ${fname}.${mcpu}.bc)

    add_custom_command(
      OUTPUT ${bc1_filename}
      COMMAND ${cu_cmd} ${file} -o ${bc1_filename}
      DEPENDS ${file} ${h_files})

    list(APPEND bc1_files ${bc1_filename})
  endforeach()

  add_custom_command(
    OUTPUT linkout.cuda.${mcpu}.bc
    COMMAND ${AOMP_BINDIR}/llvm-link ${bc1_files} -o linkout.cuda.${mcpu}.bc
    DEPENDS ${bc1_files})

  list(APPEND bc_files linkout.cuda.${mcpu}.bc)
endmacro()

set(libname "omptarget-amdgcn")

foreach(mcpu ${mcpus})
  set(bc_files)
  add_cuda_bc_library(${cuda_sources})

  set(bc_libname lib${libname}-${mcpu}.bc)
  add_custom_command(
    OUTPUT ${bc_libname}
    COMMAND ${AOMP_BINDIR}/llvm-link ${bc_files} | ${AOMP_BINDIR}/opt --always-inline -o ${OUTPUTDIR}/${bc_libname}
    DEPENDS ${bc_files})

  add_custom_target(lib${libname}-${mcpu} ALL DEPENDS ${bc_libname})

  install(FILES ${OUTPUTDIR}/${bc_libname}
     DESTINATION "${OPENMP_INSTALL_LIBDIR}/libdevice"
  )
endforeach()
