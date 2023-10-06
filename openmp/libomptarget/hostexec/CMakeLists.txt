##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build hostexec host and device  support
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  message(FATAL_ERROR "Direct configuration not supported, please use parent directory!")
endif()

# If we are bootstrapping hsa via external project we need to use find_library
# as it will not be installed.
if(DEFINED LIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH)
  find_library(HSA_RUNTIME hsa-runtime64 HINTS ${CMAKE_BINARY_DIR}/../../runtimes/rocr-runtime-prefix/src/rocr-runtime-build ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
else()
 # As of rocm-3.7, hsa is installed with cmake packages and kmt is found via hsa
  find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
endif()

if(DEFINED HSA_RUNTIME)
  libomptarget_say("Building hostexec for AMDGCN linked against libhsa")
  set(HSA_INCLUDE ${LIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH}/inc)
  set(HOSTEXEC_EXTRA_SOURCE)
  set(HOSTEXEC_DEP_LIBRARIES ${HSA_RUNTIME})
elseif(${hsa-runtime64_FOUND})
  libomptarget_say("Building hostexec for AMDGCN linked against libhsa")
  set(HSA_INCLUDE ${hsa-runtime64_DIR}/../../../include/hsa)
  set(HOSTEXEC_EXTRA_SOURCE)
  set(HOSTEXEC_DEP_LIBRARIES hsa-runtime64::hsa-runtime64)
else()
  libomptarget_say("Building hostexec for AMDGCN using dlopen for libhsa")
  set(HSA_INCLUDE ../plugins-nextgen/amdgpu/dynamic_hsa)
  set(HOSTEXEC_EXTRA_SOURCE ../plugins-nextgen/amdgpu/dynamic_hsa/hsa.cpp)
  set(HOSTEXEC_DEP_LIBRARIES)
  add_definitions(-DTARGET_NAME=AMDGPU)
endif()

   list(APPEND HOSTRPC_ARCHS "amdgcn")
   add_library(amdgcn_hostexec_services STATIC 
      services/amdgcn_hostexec.cpp 
      services/execute_service.cpp 
      services/devsanitizer.cpp 
      services/amdgcn_urilocator.cpp
      ${HOSTEXEC_EXTRA_SOURCE}
   )
   target_include_directories(
      amdgcn_hostexec_services
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/services
      ${LIBOMPTARGET_INCLUDE_DIR}
   )
   target_include_directories(
      amdgcn_hostexec_services
      PUBLIC
      ${HSA_INCLUDE}
      target_link_libraries(amdgcn_hostexec_services ${HOSTEXEC_DEP_LIBRARIES})
   )

   if(SANITIZER_AMDGPU)
      add_definitions(-DSANITIZER_AMDGPU=1)
   endif()
   add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
   set_property(TARGET amdgcn_hostexec_services PROPERTY POSITION_INDEPENDENT_CODE ON)

if (LIBOMPTARGET_DEP_CUDA_FOUND)
   list(APPEND HOSTRPC_ARCHS "nvptx")
   add_library(nvptx_hostexec_services STATIC 
      services/execute_service.cpp
   )
   target_include_directories(
      nvptx_hostexec_services
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/services
      ${LIBOMPTARGET_INCLUDE_DIR}
   )
   set_property(TARGET nvptx_hostexec_services PROPERTY POSITION_INDEPENDENT_CODE ON)
else()
   libomptarget_say("Not building hostexec for NVPTX because cuda not found")
endif()

if(NOT HOSTRPC_ARCHS)
   return()
endif()

if (LLVM_DIR)
  message("   -- Building hostexec with LLVM ${LLVM_PACKAGE_VERSION} found with CLANG_TOOL ${CLANG_TOOL}")
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(PACKAGER_TOOL clang-offload-packager PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(LINK_TOOL llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
else()
  message("   ERROR: NO LLVM FOUND! Not building hostexec.")
  return()
endif()

set(amdgpu_mcpus gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908 gfx90a gfx90c gfx940 gfx941 gfx942 gfx1010 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151)
if (DEFINED LIBOMPTARGET_AMDGCN_GFXLIST)
  set(amdgpu_mcpus ${LIBOMPTARGET_AMDGCN_GFXLIST})
endif()

set(all_capabilities 35 37 50 52 53 60 61 62 70 72 75 80 86 89 90)
set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES ${all_capabilities} CACHE STRING
  "List of CUDA Compute Capabilities to be used to compile the NVPTX DeviceRTL.")
string(TOLOWER ${LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES} LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES)
if (LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES STREQUAL "all")
  set(nvptx_sm_list ${all_capabilities})
elseif(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES STREQUAL "auto")
  if (NOT LIBOMPTARGET_DEP_CUDA_FOUND)
    libomptarget_error_say("[NVPTX] Cannot auto detect compute capability as CUDA not found.")
  endif()
  set(nvptx_sm_list ${LIBOMPTARGET_DEP_CUDA_ARCH})
else()
  string(REPLACE "," ";" nvptx_sm_list "${LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES}")
endif()
foreach(sm ${nvptx_sm_list})
   list(APPEND nvptx_mcpus "sm_${sm}")
endforeach()

set(ocl_atomics_cl_filename ${CMAKE_CURRENT_SOURCE_DIR}/src/oclAtomics.cl)
set(invoke_cpp_file_name ${CMAKE_CURRENT_SOURCE_DIR}/src/hostexec_invoke.cpp)
set(hostexec_stubs_filename ${CMAKE_CURRENT_SOURCE_DIR}/src/hostexec_stubs.cpp)
set(h_file           ${CMAKE_CURRENT_SOURCE_DIR}/src/hostexec.h)
set(internal_h_file  ${CMAKE_CURRENT_SOURCE_DIR}/src/hostexec_internal.h)

foreach(archname ${HOSTRPC_ARCHS})
   if (${archname} STREQUAL "amdgcn")
      set(triple "amdgcn-amd-amdhsa")
      set(mcpus ${amdgpu_mcpus})
      set(abi_nums "400" "500")
   endif()
   if (${archname} STREQUAL "nvptx")
      set(triple "nvptx64-nvidia-cuda")
      set(mcpus ${nvptx_mcpus})
      set(abi_nums "nvptx")
   endif()

 foreach(abi_num ${abi_nums})
   if("${abi_num}" STREQUAL "400")
     set(clang_abi_opt "-mcode-object-version=4")
   elseif("${abi_num}" STREQUAL "500")
     set(clang_abi_opt "-mcode-object-version=5")
   else()
     set(clang_abi_opt "")
   endif()
   if (${archname} STREQUAL "amdgcn")
      set(opencl_cmd ${CLANG_TOOL}
       -fvisibility=default
       -c -emit-llvm -nogpulib
       -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200
       -Dcl_khr_fp64 -Dcl_khr_fp16
       -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics
       -x cl -Xclang -cl-std=CL2.0 -Xclang -finclude-default-header
       -target amdgcn-amd-amdhsa
       ${clang_abi_opt} )
      set(ocl_atomics_cl_bc "ocl_atomics_${abi_num}-${archname}.bc")

      add_custom_target(${ocl_atomics_cl_bc}
      COMMAND ${opencl_cmd} ${ocl_atomics_cl_filename} -o ${ocl_atomics_cl_bc}
      DEPENDS ${ocl_atomics_cl_filename})
   endif()
   
   foreach(mcpu ${mcpus})
      if (${archname} STREQUAL "amdgcn")
         set(openmp_device_args 
	      -I../../runtime/src  # to pickup omp.h, we may need a dependency
	      -O${optimization_level}
	      -fopenmp -fopenmp-cuda-mode -mllvm -openmp-opt-disable 
	      ${clang_abi_opt}
	      -std=c++17 -fvisibility=hidden 
	      -fopenmp-targets=${triple} -Xopenmp-target=${triple} -march=${mcpu}
	      -c -emit-llvm --offload-device-only -nogpulib)
      endif()
      if (${archname} STREQUAL "nvptx")
         set(openmp_device_args 
	      -I../../runtime/src  # to pickup omp.h, we may need a dependency
	      -fopenmp -fopenmp-cuda-mode -mllvm -openmp-opt-disable 
	      -std=c++17 -fvisibility=hidden 
	      -fopenmp-targets=${triple} -Xopenmp-target=${triple} -march=${mcpu}
	      --cuda-feature=+ptx61,+${mcpu}
	      -c -emit-llvm --offload-device-only -nocudalib -nogpulib 
	      -Wno-unknown-cuda-version)
      endif()

      # FIXME: considre combining these two source files into singlce compiler
      #        maybe if we can get rid of compilation for opencl atomics
      set(stubs_bc_filename "hostexec-stubs-${abi_num}-${mcpu}.bc")
      add_custom_target(${stubs_bc_filename}
         COMMAND ${CLANG_TOOL} ${openmp_device_args} ${hostexec_stubs_filename} -o ${stubs_bc_filename}
         DEPENDS ${hostexec_stubs_filename} ${h_file} ${internal_h_file}
         COMMENT "Built file ${stubs_bc_filename}")

      set(hostexec_invoke_cpp_bc "hostexec-invoke-${abi_num}-${mcpu}.bc")
      add_custom_target(${hostexec_invoke_cpp_bc}
         COMMAND ${CLANG_TOOL} ${openmp_device_args} ${invoke_cpp_file_name} -o ${hostexec_invoke_cpp_bc}
	 DEPENDS ${invoke_cpp_file_name}
         COMMENT "Building bc file for hostexec_invoke: ${hostexec_invoke_cpp_bc}")

      if (NOT ${archname} STREQUAL "amdgcn")
          # for nvptx we dont need ocl_atomics
         set(ocl_atomics_cl_bc "")
      endif()

      set(libhostexec-bc "libhostexec-${abi_num}-${mcpu}.bc")
      add_custom_target(${libhostexec-bc}
         COMMAND ${LINK_TOOL} ${stubs_bc_filename} ${hostexec_invoke_cpp_bc} ${ocl_atomics_cl_bc} -o ${libhostexec-bc}
	 DEPENDS ${stubs_bc_filename} ${hostexec_invoke_cpp_bc}
         COMMENT "Building hostexec file ${libhostexec-bc}")
      add_dependencies(${libhostexec-bc} ${stubs_bc_filename})
      add_dependencies(${libhostexec-bc} ${hostexec_invoke_cpp_bc})
      if (${archname} STREQUAL "amdgcn")
         add_dependencies(${libhostexec-bc} ${ocl_atomics_cl_bc})
      endif()

      if (${archname} STREQUAL "amdgcn")
         add_dependencies(amdgcn_hostexec_services ${libhostexec-bc})
      endif()
      if (${archname} STREQUAL "nvptx")
         add_dependencies(nvptx_hostexec_services ${libhostexec-bc})
      endif()
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${libhostexec-bc} DESTINATION ${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR}/libdevice)
   endforeach() # end for each mcpu
 endforeach() # end foreach abi_num

endforeach() # end foreach archs
install(FILES ${h_file} DESTINATION ${DEVEL_PACKAGE}lib/clang/${LLVM_VERSION_MAJOR}/include)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/disable_dynamic_devmem.ll" DESTINATION ${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR})
