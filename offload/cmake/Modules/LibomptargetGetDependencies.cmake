# Try to detect in the system several dependencies required by the different
# components of libomptarget. These are the dependencies we have:
#
# libffi : required to launch target kernels given function and argument
#          pointers.

include (FindPackageHandleStandardArgs)

################################################################################
# Looking for LLVM...
################################################################################

list(APPEND LIBOMPTARGET_LLVM_INCLUDE_DIRS
  ${LLVM_MAIN_INCLUDE_DIR} ${LLVM_BINARY_DIR}/include
)

################################################################################
# Looking for libffi...
################################################################################
find_package(FFI QUIET)
set(LIBOMPTARGET_DEP_LIBFFI_FOUND ${FFI_FOUND})

################################################################################
# Looking for NVIDIA GPUs...
################################################################################
set(LIBOMPTARGET_DEP_CUDA_ARCH "sm_35")

if(TARGET nvptx-arch)
  get_property(LIBOMPTARGET_NVPTX_ARCH TARGET nvptx-arch PROPERTY LOCATION)
else()
  find_program(LIBOMPTARGET_NVPTX_ARCH NAMES nvptx-arch
               PATHS ${LLVM_TOOLS_BINARY_DIR})
endif()

if(LIBOMPTARGET_NVPTX_ARCH)
  execute_process(COMMAND ${LIBOMPTARGET_NVPTX_ARCH}
                  OUTPUT_VARIABLE LIBOMPTARGET_NVPTX_ARCH_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" ";" nvptx_arch_list "${LIBOMPTARGET_NVPTX_ARCH_OUTPUT}")
  if(nvptx_arch_list)
    set(LIBOMPTARGET_FOUND_NVIDIA_GPU TRUE)
    set(LIBOMPTARGET_NVPTX_DETECTED_ARCH_LIST "${nvptx_arch_list}")
    list(GET nvptx_arch_list 0 LIBOMPTARGET_DEP_CUDA_ARCH)
  endif()
endif()


################################################################################
# Looking for AMD GPUs...
################################################################################

if(TARGET amdgpu-arch)
  get_property(LIBOMPTARGET_AMDGPU_ARCH TARGET amdgpu-arch PROPERTY LOCATION)
else()
  find_program(LIBOMPTARGET_AMDGPU_ARCH NAMES amdgpu-arch
               PATHS ${LLVM_TOOLS_BINARY_DIR})
endif()

if(LIBOMPTARGET_AMDGPU_ARCH)
  execute_process(COMMAND ${LIBOMPTARGET_AMDGPU_ARCH}
                  OUTPUT_VARIABLE LIBOMPTARGET_AMDGPU_ARCH_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" ";" amdgpu_arch_list "${LIBOMPTARGET_AMDGPU_ARCH_OUTPUT}")
  if(amdgpu_arch_list)
    set(LIBOMPTARGET_FOUND_AMDGPU_GPU TRUE)
    set(LIBOMPTARGET_AMDGPU_DETECTED_ARCH_LIST "${amdgpu_arch_list}")
  endif()
endif()

set(OPENMP_PTHREAD_LIB ${LLVM_PTHREAD_LIB})
