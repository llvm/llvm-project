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

find_package(hsa-runtime64 1.2.0 QUIET HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm/hsa)
if (hsa-runtime64_FOUND)
   list(APPEND HOSTRPC_ARCHS "amdgcn")
   add_library(amdgcn_hostexec_services STATIC 
      services/amdgcn_hostexec.cpp 
      services/execute_service.cpp 
      services/devsanitizer.cpp 
      services/amdgcn_urilocator.cpp 
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
      ${hsa-runtime64_DIR}/../../../include)
   target_link_libraries(amdgcn_hostexec_services hsa-runtime64::hsa-runtime64)
   if(SANITIZER_AMDGPU)
      add_definitions(-DSANITIZER_AMDGPU=1)
      set(ASAN_LIB ${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/lib/linux/libclang_rt.asan-x86_64.so)
      target_link_libraries(amdgcn_hostexec_services ${ASAN_LIB})
   endif()
   set_property(TARGET amdgcn_hostexec_services PROPERTY POSITION_INDEPENDENT_CODE ON)
   set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn_hostexec_services" PARENT_SCOPE)
else()
   libomptarget_say("Not building hostexec for AMDGCN because hsa-runtime64 not found")
endif()

if( DEVICELIBS_ROOT )
  set(ROCDL_INC_OCLC ${DEVICELIBS_ROOT}/oclc/inc)
  else()
message("   ERROR: DEVICELIBS_ROOT is not set.   Not building hostexec .")
  return()
  #set(ROCDL ${CMAKE_CURRENT_SOURCE_DIR}/../../../../rocm-device-libs)
endif()



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
   set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} nvptx_hostexec_services" PARENT_SCOPE)
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

set(amdgpu_mcpus gfx700 gfx701 gfx801 gfx803 gfx900 gfx902 gfx906 gfx908 gfx90a gfx90c gfx940 gfx1010 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103)
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
   endif()
   if (${archname} STREQUAL "nvptx")
      set(triple "nvptx64-nvidia-cuda")
      set(mcpus ${nvptx_mcpus})
   endif()


   if (${archname} STREQUAL "amdgcn")
      set(opencl_cmd ${CLANG_TOOL}
       -fvisibility=default
       -c -emit-llvm -nogpulib
       -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200
       -Dcl_khr_fp64 -Dcl_khr_fp16
       -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics
       -x cl -Xclang -cl-std=CL2.0 -Xclang -finclude-default-header
       -target amdgcn-amd-amdhsa )
      set(ocl_atomics_cl_bc "ocl_atomics_${archname}.bc")
      add_custom_target(${ocl_atomics_cl_bc}
      COMMAND ${opencl_cmd} ${ocl_atomics_cl_filename} -o ${ocl_atomics_cl_bc}
      DEPENDS ${ocl_atomics_cl_filename})
   endif()

   foreach(mcpu ${mcpus})
      if (${archname} STREQUAL "amdgcn")
         set(openmp_device_args 
	      -I../../runtime/src  # to pickup omp.h, we may need a dependency
	      -I${ROCDL_INC_OCLC} -O${optimization_level}
	      -fopenmp -fopenmp-cuda-mode -mllvm -openmp-opt-disable 
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

      set(bc_filename "hostexec-stubs-${archname}-${mcpu}.bc")
      add_custom_target(${bc_filename}
         COMMAND ${CLANG_TOOL} ${openmp_device_args} ${hostexec_stubs_filename} -o ${bc_filename}
         DEPENDS ${hostexec_stubs_filename} ${h_file} ${internal_h_file}
         COMMENT "Built file ${bc_filename}")

      set(hostexec_invoke_cpp_bc "hostexec-invoke-${archname}-${mcpu}.bc")
      add_custom_target(${hostexec_invoke_cpp_bc}
         COMMAND ${CLANG_TOOL} ${openmp_device_args} ${invoke_cpp_file_name} -o ${hostexec_invoke_cpp_bc}
	 DEPENDS ${invoke_cpp_file_name}
         COMMENT "Building bc file for hostexec_invoke: ${hostexec_invoke_cpp_bc}")

      if (${archname} STREQUAL "amdgcn")

         #  amdgcn needs to compile and link in ocl_atomics
         set(invoke_resolved_bc "invoked_resolved-${archname}-${mcpu}.bc")
         add_custom_target(${invoke_resolved_bc}
            COMMAND ${LINK_TOOL} ${hostexec_invoke_cpp_bc} --internalize --only-needed ${ocl_atomics_cl_bc} -o ${invoke_resolved_bc}
	    DEPENDS ${hostexec_invoke_cpp_bc} ${ocl_atomics_cl_bc}
            COMMENT "Building invoke resolved bc ${invoke_resolved_bc}")
          add_dependencies(${invoke_resolved_bc} ${hostexec_invoke_cpp_bc} ${ocl_atomics_cl_bc})

      else()
          # for nvptx we dont need ocl_atomics
         set(invoke_resolved_bc ${hostexec_invoke_cpp_bc})
      endif()

      set(libhostexec-bc "libhostexec-${archname}-${mcpu}.bc")
      add_custom_target(${libhostexec-bc}
         COMMAND ${LINK_TOOL} ${bc_filename} --internalize --only-needed ${invoke_resolved_bc} -o ${libhostexec-bc}
	 DEPENDS ${invoke_resolved_bc} ${bc_filename} 
         COMMENT "Building hostexec file ${libhostexec-bc}")
      add_dependencies(${libhostexec-bc} ${bc_filename} ${invoke_resolved_bc})

      if (${archname} STREQUAL "amdgcn")
         add_dependencies(amdgcn_hostexec_services ${libhostexec-bc})
      endif()
      if (${archname} STREQUAL "nvptx")
         add_dependencies(nvptx_hostexec_services ${libhostexec-bc})
      endif()
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${libhostexec-bc} DESTINATION ${DEVEL_PACKAGE}lib/libdevice)
   endforeach() # end for each mcpu

endforeach() # end foreach archs
set(_version_dir 17.0.0)
install(FILES ${h_file} DESTINATION lib/clang/${_version_dir}/include)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/disable_dynamic_devmem.ll" DESTINATION ${DEVEL_PACKAGE}lib)
