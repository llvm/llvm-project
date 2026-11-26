# Try to detect in the system several dependencies required by the different
# components of libomptarget. These are the dependencies we have:

include (FindPackageHandleStandardArgs)

################################################################################
# Looking for LLVM...
################################################################################
# Note that OPENMP_STANDALONE_BUILD is FALSE, when
# openmp is built with -DLLVM_ENABLE_RUNTIMES="openmp" vs
# -DLLVM_ENABLE_PROJECTS="openmp", but openmp build
# is actually done as a standalone project build with many
# LLVM CMake variables propagated to it.
list(APPEND LIBOMPTARGET_LLVM_INCLUDE_DIRS
  ${LLVM_MAIN_INCLUDE_DIR} ${LLVM_BINARY_DIR}/include
  )
message(STATUS
  "Using LLVM include directories: ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}")

################################################################################
# Looking for offload-arch...
################################################################################
if(TARGET offload-arch)
  get_property(LIBOMPTARGET_OFFLOAD_ARCH TARGET offload-arch PROPERTY LOCATION)
else()
  find_program(LIBOMPTARGET_OFFLOAD_ARCH NAMES offload-arch
               PATHS ${LLVM_TOOLS_BINARY_DIR})
endif()

################################################################################
# Looking for NVIDIA GPUs...
################################################################################
if(LIBOMPTARGET_OFFLOAD_ARCH)
  execute_process(COMMAND ${LIBOMPTARGET_OFFLOAD_ARCH} "--only=nvptx"
                  OUTPUT_VARIABLE LIBOMPTARGET_NVPTX_ARCH_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" ";" nvptx_arch_list "${LIBOMPTARGET_NVPTX_ARCH_OUTPUT}")
  if(nvptx_arch_list)
    set(LIBOMPTARGET_FOUND_NVIDIA_GPU TRUE)
    set(LIBOMPTARGET_NVPTX_DETECTED_ARCH_LIST "${nvptx_arch_list}")
  endif()
endif()

################################################################################
# Looking for AMD GPUs...
################################################################################
if(LIBOMPTARGET_OFFLOAD_ARCH)
  execute_process(COMMAND ${LIBOMPTARGET_OFFLOAD_ARCH} "--only=amdgpu"
                  OUTPUT_VARIABLE LIBOMPTARGET_AMDGPU_ARCH_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" ";" amdgpu_arch_list "${LIBOMPTARGET_AMDGPU_ARCH_OUTPUT}")
  if(amdgpu_arch_list)
    set(LIBOMPTARGET_FOUND_AMDGPU_GPU TRUE)
    set(LIBOMPTARGET_AMDGPU_DETECTED_ARCH_LIST "${amdgpu_arch_list}")
  endif()
endif()

################################################################################
# Looking for Level0
################################################################################
if(LIBOMPTARGET_OFFLOAD_ARCH)
  execute_process(COMMAND ${LIBOMPTARGET_OFFLOAD_ARCH} "--only=intel"
                  OUTPUT_VARIABLE LIBOMPTARGET_INTELGPU_ARCH_OUTPUT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" ";" intelgpu_arch_list "${LIBOMPTARGET_INTELGPU_ARCH_OUTPUT}")
  if(intelgpu_arch_list)
    set(LIBOMPTARGET_FOUND_INTELGPU_GPU TRUE)
    set(LIBOMPTARGET_INTELGPU_DETECTED_ARCH_LIST "${intelgpu_arch_list}")
  endif()
endif()

set(OFFLOAD_PTHREAD_LIB ${LLVM_PTHREAD_LIB})
