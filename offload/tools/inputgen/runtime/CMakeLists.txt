if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20.0)
  project(InputGenGPURuntime C)
endif()

set(INPUTGEN_GPU_RT_TARGET_TRIPLES "" CACHE STRING
  "Offload target triples for InputGen GPU runtime bitcode")

set(INPUTGEN_GPU_RT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INPUTGEN_GPU_RT_INTERFACE_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../plugins-nextgen/common/include)
if(NOT INPUTGEN_GPU_RT_LLVM_INCLUDE_DIR)
  if(LLVM_MAIN_INCLUDE_DIR)
    set(INPUTGEN_GPU_RT_LLVM_INCLUDE_DIR ${LLVM_MAIN_INCLUDE_DIR})
  else()
    get_filename_component(INPUTGEN_GPU_RT_LLVM_INCLUDE_DIR
      ${CMAKE_CURRENT_SOURCE_DIR}/../../../../llvm/include ABSOLUTE)
  endif()
endif()
set_property(GLOBAL PROPERTY INPUTGEN_GPU_RT_SOURCE_DIR
             ${INPUTGEN_GPU_RT_SOURCE_DIR})
set_property(GLOBAL PROPERTY INPUTGEN_GPU_RT_INTERFACE_INCLUDE_DIR
             ${INPUTGEN_GPU_RT_INTERFACE_INCLUDE_DIR})
set_property(GLOBAL PROPERTY INPUTGEN_GPU_RT_LLVM_INCLUDE_DIR
             ${INPUTGEN_GPU_RT_LLVM_INCLUDE_DIR})

if(INPUTGEN_GPU_RT_TARGET_TRIPLES)
  set(inputgen_gpu_rt_targets ${INPUTGEN_GPU_RT_TARGET_TRIPLES})
else()
  set(inputgen_gpu_rt_targets ${LIBOMPTARGET_SYSTEM_TARGETS})
endif()

if(INPUTGEN_GPU_RT_CLANG)
elseif(LLVM_TOOLS_BINARY_DIR)
  set(INPUTGEN_GPU_RT_CLANG ${LLVM_TOOLS_BINARY_DIR}/clang${CMAKE_EXECUTABLE_SUFFIX})
elseif(CMAKE_C_COMPILER)
  set(INPUTGEN_GPU_RT_CLANG ${CMAKE_C_COMPILER})
else()
  find_program(INPUTGEN_GPU_RT_CLANG clang REQUIRED)
endif()
if(NOT INPUTGEN_GPU_RT_LLVM_LINK)
  if(LLVM_TOOLS_BINARY_DIR)
    set(INPUTGEN_GPU_RT_LLVM_LINK ${LLVM_TOOLS_BINARY_DIR}/llvm-link${CMAKE_EXECUTABLE_SUFFIX})
  else()
    find_program(INPUTGEN_GPU_RT_LLVM_LINK llvm-link REQUIRED)
  endif()
endif()

set(inputgen_gpu_rt_common_deps
  ${INPUTGEN_GPU_RT_SOURCE_DIR}/inputgen_gpu_entry_internal.h
  ${INPUTGEN_GPU_RT_SOURCE_DIR}/inputgen_gpu_instrumentor_abi.h
  ${INPUTGEN_GPU_RT_INTERFACE_INCLUDE_DIR}/InputGenInterface.hpp
  ${INPUTGEN_GPU_RT_LLVM_INCLUDE_DIR}/llvm/Frontend/Offloading/InputGenGPUABI.def)

add_custom_target(inputgen-gpu-runtime-bc)
add_custom_target(inputgen-gpu-runtime-bc-amdgpu)
add_custom_target(inputgen-gpu-runtime-bc-nvptx)

function(add_inputgen_gpu_runtime target_key compiler_triple family)
  set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/${target_key})
  set(common_flags
    -DINPUTGEN_GPU_RT_DEVICE=1
    -I${INPUTGEN_GPU_RT_SOURCE_DIR}
    -I${INPUTGEN_GPU_RT_INTERFACE_INCLUDE_DIR}
    -I${INPUTGEN_GPU_RT_LLVM_INCLUDE_DIR}
    -O3 -g -Wall -Wextra -std=c11
    --target=${compiler_triple}
    -nogpulib -nostdlibinc -fconvergent-functions -fvisibility=protected
    -flto -c -emit-llvm)

  foreach(source IN ITEMS inputgen_gpu_entry_state.c inputgen_gpu_entry_callbacks.c)
    get_filename_component(stem ${source} NAME_WE)
    add_custom_command(
      OUTPUT ${output_dir}/${stem}.bc
      COMMAND ${CMAKE_COMMAND} -E make_directory ${output_dir}
      COMMAND ${INPUTGEN_GPU_RT_CLANG} ${common_flags}
              ${INPUTGEN_GPU_RT_SOURCE_DIR}/${source} -o ${output_dir}/${stem}.bc
      DEPENDS ${INPUTGEN_GPU_RT_SOURCE_DIR}/${source} ${inputgen_gpu_rt_common_deps}
      COMMENT "Building ${source} for ${compiler_triple}"
      VERBATIM)
  endforeach()
  add_custom_command(
    OUTPUT ${output_dir}/inputgen_gpu_entry_runtime.bc
    COMMAND ${INPUTGEN_GPU_RT_LLVM_LINK}
      ${output_dir}/inputgen_gpu_entry_state.bc
      ${output_dir}/inputgen_gpu_entry_callbacks.bc
      -o ${output_dir}/inputgen_gpu_entry_runtime.bc
    DEPENDS ${output_dir}/inputgen_gpu_entry_state.bc
            ${output_dir}/inputgen_gpu_entry_callbacks.bc
    COMMENT "Linking InputGen GPU runtime bitcode for ${compiler_triple}"
    VERBATIM)
  add_custom_target(inputgen-gpu-runtime-bc-${target_key}
    DEPENDS ${output_dir}/inputgen_gpu_entry_runtime.bc)
  add_dependencies(inputgen-gpu-runtime-bc inputgen-gpu-runtime-bc-${target_key})
  add_dependencies(inputgen-gpu-runtime-bc-${family} inputgen-gpu-runtime-bc-${target_key})
  set_property(GLOBAL PROPERTY INPUTGEN_GPU_RT_${target_key}_DIR ${output_dir})
  set_property(GLOBAL PROPERTY INPUTGEN_GPU_RT_${target_key}_TRIPLE ${compiler_triple})
endfunction()

set(inputgen_gpu_rt_seen)
foreach(logical_target IN LISTS inputgen_gpu_rt_targets)
  string(REGEX REPLACE "-(JIT-)?LTO$" "" normalized_target ${logical_target})
  if(normalized_target STREQUAL "amdgpu-amd-amdhsa" OR
     normalized_target STREQUAL "amdgcn-amd-amdhsa")
    set(target_key amdgpu-amd-amdhsa)
    set(compiler_triple amdgcn-amd-amdhsa)
    set(family amdgpu)
  elseif(normalized_target STREQUAL "nvptx64-nvidia-cuda")
    set(target_key nvptx64-nvidia-cuda)
    set(compiler_triple nvptx64-nvidia-cuda)
    set(family nvptx)
  else()
    continue()
  endif()
  if(NOT target_key IN_LIST inputgen_gpu_rt_seen)
    list(APPEND inputgen_gpu_rt_seen ${target_key})
    add_inputgen_gpu_runtime(${target_key} ${compiler_triple} ${family})
  endif()
endforeach()

if(TARGET clang)
  add_dependencies(inputgen-gpu-runtime-bc clang)
endif()
if(TARGET llvm-link)
  add_dependencies(inputgen-gpu-runtime-bc llvm-link)
endif()
