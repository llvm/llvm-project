#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# We use the compiler and linker provided by the user, attempt to use the one
# used to build libomptarget or just fail.
set(LIBOMPTARGET_NVPTX_BCLIB_SUPPORTED FALSE)

if (NOT LIBOMPTARGET_NVPTX_CUDA_COMPILER STREQUAL "")
  set(LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER ${LIBOMPTARGET_NVPTX_CUDA_COMPILER})
elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
  set(LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER ${CMAKE_C_COMPILER})
else()
  return()
endif()

# Get compiler directory to try to locate a suitable linker.
get_filename_component(compiler_dir ${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER} DIRECTORY)
set(llvm_link "${compiler_dir}/llvm-link")

if (NOT LIBOMPTARGET_NVPTX_BC_LINKER STREQUAL "")
  set(LIBOMPTARGET_NVPTX_SELECTED_BC_LINKER ${LIBOMPTARGET_NVPTX_BC_LINKER})
elseif (EXISTS "${llvm_link}")
  # Use llvm-link from the compiler directory.
  set(LIBOMPTARGET_NVPTX_SELECTED_BC_LINKER "${llvm_link}")
else()
  return()
endif()

function(try_compile_bitcode output source)
  set(srcfile ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/src.cu)
  file(WRITE ${srcfile} "${source}\n")
  set(bcfile ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/out.bc)

  # The remaining arguments are the flags to be tested.
  # FIXME: Don't hardcode GPU version. This is currently required because
  #        Clang refuses to compile its default of sm_20 with CUDA 9.
  execute_process(
    COMMAND ${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER} ${ARGN}
      --cuda-gpu-arch=sm_35 -c ${srcfile} -o ${bcfile}
    RESULT_VARIABLE result
    OUTPUT_QUIET ERROR_QUIET)
  if (result EQUAL 0)
    set(${output} TRUE PARENT_SCOPE)
  else()
    set(${output} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Save for which compiler we are going to do the following checks so that we
# can discard cached values if the user specifies a different value.
set(discard_cached FALSE)
if (DEFINED LIBOMPTARGET_NVPTX_CHECKED_CUDA_COMPILER AND
    NOT("${LIBOMPTARGET_NVPTX_CHECKED_CUDA_COMPILER}" STREQUAL "${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER}"))
  set(discard_cached TRUE)
endif()
set(LIBOMPTARGET_NVPTX_CHECKED_CUDA_COMPILER "${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER}" CACHE INTERNAL "" FORCE)

function(check_bitcode_compilation output source)
  if (${discard_cached} OR NOT DEFINED ${output})
    message(STATUS "Performing Test ${output}")
    # Forward additional arguments which contain the flags.
    try_compile_bitcode(result "${source}" ${ARGN})
    set(${output} ${result} CACHE INTERNAL "" FORCE)
    if(${result})
      message(STATUS "Performing Test ${output} - Success")
    else()
      message(STATUS "Performing Test ${output} - Failed")
    endif()
  endif()
endfunction()

# These flags are required to emit LLVM Bitcode. We check them together because
# if any of them are not supported, there is no point in finding out which are.
set(compiler_flags_required -emit-llvm -O1 --cuda-device-only --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
set(compiler_flags_required_src "extern \"C\" __device__ int thread() { return threadIdx.x; }")
check_bitcode_compilation(LIBOMPTARGET_NVPTX_CUDA_COMPILER_SUPPORTS_FLAGS_REQUIRED "${compiler_flags_required_src}" ${compiler_flags_required})

# It makes no sense to continue given that the compiler doesn't support
# emitting basic LLVM Bitcode
if (NOT LIBOMPTARGET_NVPTX_CUDA_COMPILER_SUPPORTS_FLAGS_REQUIRED)
  return()
endif()

set(LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER_FLAGS ${compiler_flags_required})

# Declaring external shared device variables might need an additional flag
# since Clang 7.0 and was entirely unsupported since version 4.0.
set(extern_device_shared_src "extern __device__ __shared__ int test;")

check_bitcode_compilation(LIBOMPTARGET_NVPTX_CUDA_COMPILER_SUPPORTS_EXTERN_SHARED "${extern_device_shared_src}" ${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER_FLAGS})
if (NOT LIBOMPTARGET_NVPTX_CUDA_COMPILER_SUPPORTS_EXTERN_SHARED)
  set(compiler_flag_fcuda_rdc -fcuda-rdc)
  set(compiler_flag_fcuda_rdc_full ${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER_FLAGS} ${compiler_flag_fcuda_rdc})
  check_bitcode_compilation(LIBOMPTARGET_NVPTX_CUDA_COMPILER_SUPPORTS_FCUDA_RDC "${extern_device_shared_src}" ${compiler_flag_fcuda_rdc_full})

  if (NOT LIBOMPTARGET_NVPTX_CUDA_COMPILER_SUPPORTS_FCUDA_RDC)
    return()
  endif()

  set(LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER_FLAGS "${compiler_flag_fcuda_rdc_full}")
endif()

# We can compile LLVM Bitcode from CUDA source code!
set(LIBOMPTARGET_NVPTX_BCLIB_SUPPORTED TRUE)
