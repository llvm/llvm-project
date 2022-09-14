##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
##===----------------------------------------------------------------------===##
#
# llvm-project/openmp/libomptarget/hostrpc/CMakeLists.txt
# Written by Greg Rodgers (Gregory.Rodgers@amd.com)
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
add_subdirectory(services)
# find_package(LLVM is done in libomptarget/CMakeLists.txt
if (LLVM_DIR)
	message("   -- Building hostrpc with LLVM ${LLVM_PACKAGE_VERSION} found with CLANG_TOOL ${CLANG_TOOL}")
else()
  message("   ERROR: NO LLVM FOUND! Not building hostrpc .")
  return()
endif()

if( DEVICELIBS_ROOT )
  set(ROCDL_INC_OCKL ${DEVICELIBS_ROOT}/ockl/inc)
  set(ROCDL_INC_OCLC ${DEVICELIBS_ROOT}/oclc/inc)
  set(ROCDL_INC_OCML ${DEVICELIBS_ROOT}/ocml/inc)
  set(ROCDL_INC_IRIF ${DEVICELIBS_ROOT}/irif/inc)
else()
  message("   ERROR: DEVICELIBS_ROOT is not set.   Not building hostrpc .")
  return()
  #set(ROCDL ${CMAKE_CURRENT_SOURCE_DIR}/../../../../rocm-device-libs)
endif()

set(amdgpu_mcpus gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908 gfx90a gfx90c gfx940 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036)
if (DEFINED LIBOMPTARGET_AMDGCN_GFXLIST)
  set(amdgpu_mcpus ${LIBOMPTARGET_AMDGCN_GFXLIST})
endif()
set(triple "amdgcn-amd-amdhsa")
set(archname "amdgcn")

# see if this build is for LLVM_ENABLE_RUNTIMES='openmp'
set(_xdir "")
foreach(proj ${LLVM_ENABLE_RUNTIMES})
  string(TOUPPER "${proj}" canon_name)
  if ("${canon_name}" STREQUAL "OPENMP")
     set(_xdir "/openmp")
  endif()
endforeach()

set(cl_file_name ${CMAKE_CURRENT_SOURCE_DIR}/src/hostrpc_invoke.cl)
set(cpp_file_name ${CMAKE_CURRENT_SOURCE_DIR}/src/hostrpc.cpp)
file(GLOB h_files "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
  # Eventually we should get hostrpc for other archs, for now just amdgcn
foreach(mcpu ${amdgpu_mcpus})
  set(bc_cl_filename "hostrpc_invoke-${archname}-${mcpu}.bc")
  set(obj_filename "hostrpc-${archname}-${mcpu}.o")
  set(opencl_cmd ${CLANG_TOOL}
    -fvisibility=default
    -S -nogpulib -emit-llvm
    -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200
    -Dcl_khr_fp64 -Dcl_khr_fp16
    -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics
    -x cl -Xclang -cl-std=CL2.0 -Xclang -finclude-default-header
    -target amdgcn-amd-amdhsa
    -I${ROCDL_INC_OCKL}
    -I${ROCDL_INC_OCLC}
    -I${ROCDL_INC_OCML}
    -I${ROCDL_INC_IRIF}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src)
    #-I${LLVM_COMPILER}/include
    #-I${ROCM_DIR}/include)
  add_custom_command(
       OUTPUT ${bc_cl_filename}
       COMMAND ${opencl_cmd} ${cl_file_name} -o ${bc_cl_filename}
       DEPENDS ${cl_file_name} ${h_files})
  # Compile hostrpc.cpp using openmp clang++ -c into heterogeneous object
  set(openmp_args -O${optimization_level}
    -fopenmp
    -fopenmp-targets=${triple}
    -Xopenmp-target=${triple}
    -march=${mcpu}
    -nogpulib
    -c
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${CMAKE_BINARY_DIR}${_xdir}/runtime/src)
  set(bc_cpp_filename "hostrpc-${archname}-${mcpu}.bc")
  add_custom_command(
     OUTPUT ${obj_filename}
     COMMAND ${CLANG_TOOL} ${openmp_args} ${cpp_file_name} -o ${obj_filename}
     DEPENDS ${cpp_file_name} ${h_files} ${clang_driver_runtime_depends})
  # extract device bc from the heterogeneous object file
  add_custom_command(
      OUTPUT ${bc_cpp_filename}
      COMMAND ${CLANG_OFFLOAD_BUNDLER_TOOL} -type=o -targets=openmp-${triple}-${mcpu} -input=${obj_filename} -output=${bc_cpp_filename} -unbundle
      DEPENDS ${obj_filename} ${clang_driver_runtime_depends})

  set(libhostrpc-bc ${CMAKE_BINARY_DIR}/libhostrpc-amdgcn-${mcpu}.bc)
  add_custom_command(
    OUTPUT ${libhostrpc-bc}
    COMMAND ${LINK_TOOL} ${bc_cpp_filename} ${bc_cl_filename} -o ${libhostrpc-bc}
    DEPENDS ${bc_cpp_filename} ${bc_cl_filename}
    COMMENT "Built hostrpc file ${libhostrpc-bc}")
  add_custom_target(libhostrpc-target-${mcpu} ALL DEPENDS ${libhostrpc-bc})
  # Add a file-level dependency to ensure that llvm-link is up-to-date
  # By default, add_custom_command only builds the tool if the executable is missing
  if("${LINK_TOOL}" STREQUAL "$<TARGET_FILE:llvm-link>")
    add_custom_command(OUTPUT ${libhostrpc-bc}
      DEPENDS llvm-link
      APPEND)
  endif()
endforeach()

#set(triple "nvptx-nvidia-cuda")
# Since libomptarget includes hostrpc, we only need to install hostrpc headers
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/hostrpc.h" DESTINATION ${DEVEL_PACKAGE}include)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/disable_dynamic_devmem.ll" DESTINATION ${DEVEL_PACKAGE}lib)
