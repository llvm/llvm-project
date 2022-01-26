##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build the AMDGCN Device RTL bitcode library using clang -ffreestanding
#
##===----------------------------------------------------------------------===##

set(LIBOMPTARGET_BUILD_AMDGCN_BCLIB FALSE CACHE BOOL
  "Can be set to true to enable building this library.")

if (NOT LIBOMPTARGET_BUILD_AMDGCN_BCLIB)
  libomptarget_say("Not building AMDGCN device RTL: Disabled by LIBOMPTARGET_BUILD_AMDGCN_BCLIB")
  return()
endif()

if (NOT LIBOMPTARGET_LLVM_INCLUDE_DIRS)
  libomptarget_say("Not building AMDGCN device RTL: Missing definition for LIBOMPTARGET_LLVM_INCLUDE_DIRS")
  return()
endif()


# Copied from nvptx CMakeLists
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(aux_triple x86_64-unknown-linux-gnu)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "ppc64le")
  set(aux_triple powerpc64le-unknown-linux-gnu)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(aux_triple aarch64-unknown-linux-gnu)
else()
  libomptarget_say("Not building AMDGCN device RTL: unknown host arch: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
  return()
endif()

if (LLVM_DIR)
  # Builds that use pre-installed LLVM have LLVM_DIR set.
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(LINK_TOOL llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(OPT_TOOL opt PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  if ((NOT CLANG_TOOL) OR (NOT LINK_TOOL) OR (NOT OPT_TOOL))
    libomptarget_say("Not building AMDGCN device RTL. Missing clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL} or opt: ${OPT_TOOL}")
    return()
  else()
    libomptarget_say("Building AMDGCN device RTL. Using clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL} and opt: ${OPT_TOOL}")
  endif()
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING AND NOT OPENMP_STANDALONE_BUILD)
  # LLVM in-tree builds may use CMake target names to discover the tools.
  set(CLANG_TOOL $<TARGET_FILE:clang>)
  set(LINK_TOOL $<TARGET_FILE:llvm-link>)
  set(OPT_TOOL $<TARGET_FILE:opt>)
  libomptarget_say("Building AMDGCN device RTL. Using clang from in-tree build")
else()
  libomptarget_say("Not building AMDGCN device RTL. No appropriate clang found")
  return()
endif()

project(omptarget-amdgcn)

add_custom_target(omptarget-amdgcn ALL)

#optimization level
set(optimization_level 2)

# Activate RTL message dumps if requested by the user.
if(LIBOMPTARGET_NVPTX_DEBUG)
  set(CUDA_DEBUG -DOMPTARGET_NVPTX_DEBUG=-1 -g)
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
  ${devicertl_base_directory}/common/src/shuffle.cpp
  ${devicertl_base_directory}/common/src/sync.cu
  ${devicertl_base_directory}/common/src/task.cu)

set(h_files
  ${CMAKE_CURRENT_SOURCE_DIR}/src/amdgcn_interface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/target_impl.h
  ${devicertl_base_directory}/common/debug.h
  ${devicertl_base_directory}/common/omptarget.h
  ${devicertl_base_directory}/common/omptargeti.h
  ${devicertl_base_directory}/common/state-queue.h
  ${devicertl_base_directory}/common/state-queuei.h
  ${devicertl_base_directory}/common/support.h)

# for both in-tree and out-of-tree build
if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(OUTPUTDIR ${CMAKE_CURRENT_BINARY_DIR})
else()
  set(OUTPUTDIR ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
endif()

# create gfx bitcode libraries
set(mcpus gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908 gfx90a gfx1010 gfx1030 gfx1031)
if (DEFINED LIBOMPTARGET_AMDGCN_GFXLIST)
  set(mcpus ${LIBOMPTARGET_AMDGCN_GFXLIST})
endif()

# Prepend -I to each list element
set (LIBOMPTARGET_LLVM_INCLUDE_DIRS_AMDGCN "${LIBOMPTARGET_LLVM_INCLUDE_DIRS}")
list(TRANSFORM LIBOMPTARGET_LLVM_INCLUDE_DIRS_AMDGCN PREPEND "-I")

macro(add_cuda_bc_library)
  set(cu_cmd ${CLANG_TOOL}
    -xc++
    -c
    -mllvm -openmp-opt-disable
    -std=c++14
    -ffreestanding
    -target amdgcn-amd-amdhsa
    -emit-llvm
    -Xclang -aux-triple -Xclang ${aux_triple}
    -fopenmp -fopenmp-cuda-mode -Xclang -fopenmp-is-device
    -D__AMDGCN__
    -Xclang -target-cpu -Xclang ${mcpu}
    -fvisibility=hidden
    -Wno-unused-value
    -nogpulib
    -O${optimization_level}
    ${CUDA_DEBUG}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${devicertl_base_directory}/common/include
    -I${devicertl_base_directory}
    -I${devicertl_base_directory}/../include
    ${LIBOMPTARGET_LLVM_INCLUDE_DIRS_AMDGCN})

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
    COMMAND ${LINK_TOOL} ${bc1_files} -o linkout.cuda.${mcpu}.bc
    DEPENDS ${bc1_files})

  list(APPEND bc_files linkout.cuda.${mcpu}.bc)
endmacro()

set(libname "omptarget-amdgcn")

set(toolchain_deps "")
if(TARGET llvm-link)
  list(APPEND toolchain_deps llvm-link)
endif()
if(TARGET opt)
  list(APPEND toolchain_deps opt)
endif()

foreach(mcpu ${mcpus})
  set(bc_files)
  add_cuda_bc_library(${cuda_sources})

  set(bc_libname lib${libname}-${mcpu}.bc)
  add_custom_command(
    OUTPUT ${bc_libname}
    COMMAND ${LINK_TOOL} ${bc_files} | ${OPT_TOOL} --always-inline -o ${OUTPUTDIR}/${bc_libname}
    DEPENDS ${bc_files} ${toolchain_deps})

  add_custom_target(lib${libname}-${mcpu} ALL DEPENDS ${bc_libname})

  install(FILES ${OUTPUTDIR}/${bc_libname}
     DESTINATION "${OPENMP_INSTALL_LIBDIR}"
  )
endforeach()
