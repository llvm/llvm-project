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
if(OFFLOAD_EXTERNAL_PROJECT_UNIFIED_ROCR)
  # With ROCm 6.3 ROCr and ROCt are in one repo and we need to account for the different paths
  find_library(HSA_RUNTIME hsa-runtime64
    HINTS
    ${CMAKE_BINARY_DIR}/../../runtimes/rocr-runtime-prefix/src/rocr-runtime-build/rocr/lib
    ${CMAKE_INSTALL_PREFIX}
    PATHS
    /opt/rocm)
else()
  if(DEFINED LIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH)
    find_library(HSA_RUNTIME hsa-runtime64
      HINTS
      ${CMAKE_BINARY_DIR}/../../runtimes/rocr-runtime-prefix/src/rocr-runtime-build
      ${CMAKE_INSTALL_PREFIX}
      PATHS
      /opt/rocm)
  else()
    # As of rocm-3.7, hsa is installed with cmake packages and kmt is found via hsa
    find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
  endif()
endif()

if(DEFINED HSA_RUNTIME)
  message(STATUS "Building hostexec for AMDGCN linked against libhsa")
  if(OFFLOAD_EXTERNAL_PROJECT_UNIFIED_ROCR)
    set(HSA_INCLUDE ${LIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH}/runtime/hsa-runtime/inc)
  else()
    set(HSA_INCLUDE ${LIBOMPTARGET_EXTERNAL_PROJECT_HSA_PATH}/inc)
  endif()
  set(HOSTEXEC_EXTRA_SOURCE)
  set(HOSTEXEC_DEP_LIBRARIES ${HSA_RUNTIME})
  message(STATUS "HSA Runtime found: ${HSA_RUNTIME}")
  message(STATUS "HSA Runtime include: ${HSA_INCLUDE}")
elseif(${hsa-runtime64_FOUND})
  message(STATUS "Building hostexec for AMDGCN linked against libhsa64")
  set(HSA_INCLUDE ${hsa-runtime64_DIR}/../../../include/hsa)
  get_target_property(HSA_INCLUDE hsa-runtime64::hsa-runtime64 INTERFACE_INCLUDE_DIRECTORIES)
  set(HOSTEXEC_EXTRA_SOURCE)
  set(HOSTEXEC_DEP_LIBRARIES hsa-runtime64::hsa-runtime64)
else()
  message(STATUS "Building hostexec for AMDGCN using dlopen for libhsa")
  set(HSA_INCLUDE ../plugins-nextgen/amdgpu/dynamic_hsa)
  set(HOSTEXEC_EXTRA_SOURCE ../plugins-nextgen/amdgpu/dynamic_hsa/hsa.cpp)
  set(HOSTEXEC_DEP_LIBRARIES)
  add_definitions(-DTARGET_NAME=AMDGPU)
endif()

list(APPEND HOSTRPC_ARCHS "amdgpu")
add_library(amdgpu_hostexec_services STATIC 
  services/amdgpu_hostexec.cpp 
  services/execute_service.cpp 
  services/devsanitizer.cpp 
  services/amdgpu_urilocator.cpp
  ${HOSTEXEC_EXTRA_SOURCE}
)
target_include_directories(
  amdgpu_hostexec_services
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/services
  ${LIBOMPTARGET_INCLUDE_DIR}
)
target_include_directories(
  amdgpu_hostexec_services
  PUBLIC
  ${HSA_INCLUDE}
  target_link_libraries(amdgpu_hostexec_services ${HOSTEXEC_DEP_LIBRARIES})
)

add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
set_property(TARGET amdgpu_hostexec_services PROPERTY POSITION_INDEPENDENT_CODE ON)

option(LIBOMPTARGET_BUILD_DEVICE_FORTRT "Build device fortran IO runtime to use hostexec" OFF)
if (LIBOMPTARGET_BUILD_DEVICE_FORTRT)
 add_compile_definitions(_HOSTEXEC_BUILD_DEVICE_FORTRT)
 target_link_libraries(amdgpu_hostexec_services FortranDecimal -L${CMAKE_BINARY_DIR}/../../lib  -L${CMAKE_INSTALL_PREFIX}/lib)
 target_link_libraries(amdgpu_hostexec_services FortranRuntime -L${CMAKE_BINARY_DIR}/../../lib  -L${CMAKE_INSTALL_PREFIX}/lib)
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
else()
   message(STATUS "Not building hostexec for NVPTX because cuda not found")
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

set(ocl_atomics_cl_filename ${CMAKE_CURRENT_SOURCE_DIR}/src/oclAtomics.cl)
set(invoke_cpp_file_name ${CMAKE_CURRENT_SOURCE_DIR}/src/hostexec_invoke.cpp)
set(hostexec_stubs_filename ${CMAKE_CURRENT_SOURCE_DIR}/src/hostexec_stubs.cpp)
set(h_file           ${CMAKE_CURRENT_SOURCE_DIR}/src/hostexec.h)
set(internal_h_file  ${CMAKE_CURRENT_SOURCE_DIR}/src/hostexec_internal.h)

foreach(archname ${HOSTRPC_ARCHS})
   if (${archname} STREQUAL "amdgpu")
      set(triple "amdgcn-amd-amdhsa")
   endif()
   if (${archname} STREQUAL "nvptx")
      set(triple "nvptx64-nvidia-cuda")
   endif()

   if (${archname} STREQUAL "amdgpu")
      set(opencl_cmd ${CLANG_TOOL}
       -fvisibility=default
       -c -emit-llvm -nogpulib -nogpuinc
       -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200
       -Dcl_khr_fp64 -Dcl_khr_fp16
       -Dcl_khr_subgroups -Dcl_khr_int64_base_atomics -Dcl_khr_int64_extended_atomics
       -x cl -Xclang -cl-std=CL2.0 -Xclang -finclude-default-header
       -Xclang -mcode-object-version=none
       -target amdgcn-amd-amdhsa )
      set(ocl_atomics_cl_bc "ocl_atomics_${archname}.bc")

      add_custom_target(${ocl_atomics_cl_bc}
      COMMAND ${opencl_cmd} ${ocl_atomics_cl_filename} -o ${ocl_atomics_cl_bc}
      DEPENDS ${ocl_atomics_cl_filename})
   endif()

    if (${archname} STREQUAL "amdgpu")
      find_package(AMDDeviceLibs REQUIRED CONFIG
                   HINTS ${CMAKE_BINARY_DIR}/../../tools/rocm-device-libs
                         ${CMAKE_BINARY_DIR}/../rocm-device-libs-prefix/src/rocm-device-libs-build
                         ${CMAKE_INSTALL_PREFIX}
                   PATHS /opt/rocm
      )
      get_target_property(ocml_path ocml IMPORTED_LOCATION)
      get_target_property(ockl_path ockl IMPORTED_LOCATION)

      set(openmp_device_args
          -I../../runtime/src  # to pickup omp.h, we may need a dependency
          -I${CMAKE_CURRENT_SOURCE_DIR}/../DeviceRTL/include # pickup Platform.h
          -O${optimization_level}
          -fopenmp -fopenmp-cuda-mode -mllvm -openmp-opt-disable
          -std=c++17 -fvisibility=hidden -nogpuinc
          -fopenmp-targets=${triple} -Xopenmp-target=${triple} -march=
          -Xclang -mcode-object-version=none
          -Xclang -mlink-builtin-bitcode -Xclang ${ocml_path}
          -Xclang -mlink-builtin-bitcode -Xclang ${ockl_path}
          -Wno-linker-warnings # Silence the empty host compilation.
          -c -emit-llvm --offload-device-only -nogpulib)
    elseif(${archname} STREQUAL "nvptx")
       set(openmp_device_args 
      -I../../runtime/src  # to pickup omp.h, we may need a dependency
      -I${CMAKE_CURRENT_SOURCE_DIR}/../DeviceRTL/include # pickup Platform.h
      -fopenmp -fopenmp-cuda-mode -mllvm -openmp-opt-disable 
      -std=c++17 -fvisibility=hidden 
      -fopenmp-targets=${triple} -Xopenmp-target=${triple} -march=
      --cuda-feature=+ptx61 -nogpuinc
      -c -emit-llvm --offload-device-only -nocudalib -nogpulib 
      -Wno-unknown-cuda-version)
    endif()

    # FIXME: considre combining these two source files into singlce compiler
    #        maybe if we can get rid of compilation for opencl atomics
    set(stubs_bc_filename "hostexec-stubs-${archname}.bc")
    add_custom_target(${stubs_bc_filename}
       COMMAND ${CLANG_TOOL} ${openmp_device_args} ${hostexec_stubs_filename} -o ${stubs_bc_filename}
       DEPENDS ${hostexec_stubs_filename} ${h_file} ${internal_h_file}
       COMMENT "Built file ${stubs_bc_filename}")

    set(hostexec_invoke_cpp_bc "hostexec-invoke-${archname}.bc")
    add_custom_target(${hostexec_invoke_cpp_bc}
       COMMAND ${CLANG_TOOL} ${openmp_device_args} ${invoke_cpp_file_name} -o ${hostexec_invoke_cpp_bc}
 DEPENDS ${invoke_cpp_file_name}
       COMMENT "Building bc file for hostexec_invoke: ${hostexec_invoke_cpp_bc}")

    if (NOT ${archname} STREQUAL "amdgpu")
        # for nvptx we dont need ocl_atomics
       set(ocl_atomics_cl_bc "")
    endif()

    set(libhostexec-bc "libhostexec-${archname}.bc")
    set(libhostexec-bc-preprep "libhostexec-${archname}-preprep.bc")
    add_custom_target(${libhostexec-bc-preprep}
       COMMAND ${LINK_TOOL} ${stubs_bc_filename} ${hostexec_invoke_cpp_bc} ${ocl_atomics_cl_bc} -o ${libhostexec-bc-preprep}
 DEPENDS ${stubs_bc_filename} ${hostexec_invoke_cpp_bc}
       COMMENT "Building hostexec file ${libhostexec-bc-preprep}")
    add_dependencies(${libhostexec-bc-preprep} ${stubs_bc_filename})
    add_dependencies(${libhostexec-bc-preprep} ${hostexec_invoke_cpp_bc})

    # Run the prep tool on the library to replace internal attribute with linkonce_odr for dm_alloc only.
    add_custom_target(${libhostexec-bc}
      COMMAND ${PREP_TOOL} -dm ${libhostexec-bc-preprep}
                    -o ${libhostexec-bc}
      DEPENDS ${libhostexec-bc-preprep}
      COMMENT "Running ${PREP_TOOL} for ${libhostexec-bc}")
    add_dependencies(${libhostexec-bc} prep-libomptarget-bc)
    if (${archname} STREQUAL "amdgpu")
       add_dependencies(${libhostexec-bc} ${ocl_atomics_cl_bc})
    endif()

    if (${archname} STREQUAL "amdgpu")
       add_dependencies(amdgpu_hostexec_services ${libhostexec-bc})
    endif()
    if (${archname} STREQUAL "nvptx")
       add_dependencies(nvptx_hostexec_services ${libhostexec-bc})
    endif()
endforeach() # end foreach archs

install(FILES ${h_file} DESTINATION lib/clang/${LLVM_VERSION_MAJOR}/include)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/disable_dynamic_devmem.ll" DESTINATION ${OPENMP_INSTALL_LIBDIR})
