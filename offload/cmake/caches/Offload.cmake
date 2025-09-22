set(LLVM_ENABLE_PROJECTS "clang;clang-tools-extra;lld" CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES "compiler-rt;libunwind;libcxx;libcxxabi;openmp;offload" CACHE STRING "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE BOOL "")

set(LLVM_RUNTIME_TARGETS default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda CACHE STRING "") 
set(RUNTIMES_nvptx64-nvidia-cuda_CACHE_FILES "${CMAKE_SOURCE_DIR}/../libcxx/cmake/caches/NVPTX.cmake" CACHE STRING "")
set(RUNTIMES_amdgcn-amd-amdhsa_CACHE_FILES "${CMAKE_SOURCE_DIR}/../libcxx/cmake/caches/AMDGPU.cmake" CACHE STRING "")
set(RUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES "compiler-rt;libc;openmp;libcxx;libcxxabi" CACHE STRING "")
set(RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES "compiler-rt;libc;openmp;libcxx;libcxxabi" CACHE STRING "")
