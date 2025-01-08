# This file is meant for test builds on one basic AMDGPU buildbot only.

# Install directory set to /tmp as this is a bot config
set(CMAKE_INSTALL_PREFIX /tmp/llvm.install.test CACHE STRING "")

set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES "compiler-rt;openmp;offload" CACHE STRING "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_LIT_ARGS "-v --show-unsupported --timeout 100 --show-xfail -j 32" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "host;AMDGPU" CACHE STRING "")

set(CLANG_DEFAULT_LINKER "lld" CACHE STRING "")

set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_C_COMPILER_LAUNCHER ccache CACHE STRING "")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache CACHE STRING "")
