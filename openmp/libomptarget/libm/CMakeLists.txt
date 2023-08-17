##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# libomptarget/libm/libdevice/CMakeLists.txt
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  project(libm-project)
  message("--------------------------------------------")
endif()

# find_package(LLVM done in libomptarget/CMakeLists.txt
if (LLVM_DIR)
  message("   -- Building libm bitcodes with LLVM ${LLVM_PACKAGE_VERSION} using ${CLANG_TOOL}")
else()
  message("   ERROR: NO LLVM FOUND! Not building libm libdevice.")
  return()
endif()

#optimization level
set(optimization_level 2)

# Get list of AMD GPUs to build for
set(amdgpu_mcpus gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908 gfx90a gfx90c gfx940 gfx941 gfx942 gfx1010 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151)
if (DEFINED LIBOMPTARGET_AMDGCN_GFXLIST)
  set(amdgpu_mcpus ${LIBOMPTARGET_AMDGCN_GFXLIST})
endif()

# see if this build is for LLVM_ENABLE_RUNTIMES='openmp'
set(_xdir "")
foreach(proj ${LLVM_ENABLE_RUNTIMES})
  string(TOUPPER "${proj}" canon_name)
  if ("${canon_name}" STREQUAL "OPENMP")
     set(_xdir "/openmp")
  endif()
endforeach()

# prepare variables used by build_static_device_bc_lib
set(triple "amdgcn-amd-amdhsa")
set(systemarch "amdgcn")

foreach(mcpu ${amdgpu_mcpus})
  set(cpu_target x86_64-pc-linux-gnu)
  if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
    set(cpu_target powerpc64le-linux-gnu)
  endif()
  # Need omp.h from the build directory
  set(omp_common_args
    -c 
    -I ${CMAKE_BINARY_DIR}${_xdir}/runtime/src
    -emit-llvm
    -target ${cpu_target}
    -fopenmp 
    -fopenmp-targets=${triple}
    -Xopenmp-target=${triple}
    -march=${mcpu}
    --cuda-device-only
    -nocudalib
    -Xclang -mcode-object-version=none
    -O${optimization_level} )
  set(libm-bc ${CMAKE_BINARY_DIR}/libm-${mcpu}.bc)
  set(openmp_c_cmd ${CLANG_TOOL} ${omp_common_args}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/libm.c)
  add_custom_command(
    OUTPUT ${libm-bc}
    COMMAND ${openmp_c_cmd} -o ${libm-bc}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/libm.c"
    COMMENT "Built libm file ${libm-bc}")
  add_custom_target(libm-target-${mcpu} ALL DEPENDS ${libm-bc})
  if("${CLANG_TOOL}" STREQUAL "$<TARGET_FILE:clang")
    add_custom_command(OUTPUT ${libm-bc}
      DEPENDS clang
      APPEND)
  endif()
endforeach()

#set(nvptx_numbers $ENV{NVPTXGPUS})
#if(NOT nvptx_numbers)
	#set(nvptx_numbers "35,37,50,52,53,60,61,62,70,72,75,80,86")
  #endif()
#set(nvptxgpus)
#string(REGEX REPLACE "," ";" nvptx_numbers_list ${nvptx_numbers})
#foreach(sm_number ${nvptx_numbers_list})
	#list(APPEND nvptxgpus "sm_${sm_number}")
   #endforeach()
#set(gpulist ${nvptxgpus})
#set(systemarch "nvptx")
#set(triple "nvptx64-nvidia-cuda ")
#build_static_device_bc_lib()

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  message("--------------------------------------------")
endif()
