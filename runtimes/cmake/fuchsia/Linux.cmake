include(${CMAKE_CURRENT_LIST_DIR}/Common.cmake)

# This file sets up the builtins and runtimes for the Linux platform for
# a Fuchsia toolchain build. It can be used in an end to end toolchain build,
# or be used standalone.

# If build end2end. Expect FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_MODE set to END2END
# And FUCHSIA_TOOLCHAIN_LINUX_RUNTIMES_TARGETS to contain a list of targets
# if build standalone. Expect FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_MODE set to BUILTIN or RUNTIMES
# and FUCHSIA_TOOLCHAIN_LINUX_RUNTIMES_TARGETS to contain a single target triple.

# Due to a limitation in compiler-rt and runtimes cmake configuration, both
# depends on the cmake files under LLVM's build directory. Therefore, in order to
# build runtimes in standalone mode, the LLVM+Clang needs to be built first.

# Example of building a x86_64-unknown-linux-gnu builtins:

# cmake \
#   -GNinja \
#   -DCMAKE_MAKE_PROGRAM=ninja \
#   -DCMAKE_INSTALL_PREFIX="" \
#   -DCMAKE_C_COMPILER=LLVM_CLANG_BUILD_DIR/bin/clang \             
#   -DCMAKE_CXX_COMPILER=LLVM_CLANG_BUILD_DIR/bin/clang++ \             
#   -DCMAKE_ASM_COMPILER=LLVM_CLANG_BUILD_DIR/bin/clang \             
#   -DCLANG_REPOSITORY_STRING="https://llvm.googlesource.com/llvm-project" \
#   -DCMAKE_AR=LLVM_CLANG_BUILD_DIR/bin/llvm-ar \             
#   -DCMAKE_LINKER=LLVM_CLANG_BUILD_DIR/bin/ld.lld \             
#   -DCMAKE_NM=LLVM_CLANG_BUILD_DIR/bin/llvm-nm \             
#   -DCMAKE_OBJCOPY=LLVM_CLANG_BUILD_DIR/bin/llvm-objcopy \             
#   -DCMAKE_OBJDUMP=LLVM_CLANG_BUILD_DIR/bin/llvm-objdump \             
#   -DCMAKE_RANLIB=LLVM_CLANG_BUILD_DIR/bin/llvm-ranlib \             
#   -DCMAKE_READELF=LLVM_CLANG_BUILD_DIR/bin/llvm-readelf \             
#   -DCMAKE_STRIP=LLVM_CLANG_BUILD_DIR/bin/llvm-strip \             
#   -DCMAKE_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_aarch64-unknown-linux-gnu_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_armv7-unknown-linux-gnueabihf_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_i386-unknown-linux-gnu_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_riscv64-unknown-linux-gnu_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_x86_64-unknown-linux-gnu_SYSROOT=LINUX_SYSROOT_PATH \
#   -DFUCHSIA_SDK=/mnt/nvme_crypt/SRC/llvm-prebuilts/fuchsia-idk \
#   -DLLVM_ENABLE_LTO=False \
#   -DLLVM_ENABLE_ASSERTIONS=True \
#   -DLLVM_ENABLE_BACKTRACES=True \                                            
#   -DFUCHSIA_TOOLCHAIN_RUNTIME_BUILD_MODE=BUILTIN \                         
#   -DFUCHSIA_TOOLCHAIN_LINUX_RUNTIMES_TARGETS="x86_64-unknown-linux-gnu" \
#   -DLLVM_CMAKE_DIR=LLVM_CLANG_BUILD_DIR/lib/cmake/llvm \
#   -C ../runtimes/cmake/fuchsia/Linux.cmake ../compiler-rt/lib/builtins

# Example of building a x86_64-unknown-linux-gnu runtimes:

# cmake \
#   -GNinja \
#   -DCMAKE_MAKE_PROGRAM=ninja \
#   -DCMAKE_INSTALL_PREFIX="" \
#   -DCMAKE_C_COMPILER=LLVM_CLANG_BUILD_DIR/bin/clang \             
#   -DCMAKE_CXX_COMPILER=LLVM_CLANG_BUILD_DIR/bin/clang++ \             
#   -DCMAKE_ASM_COMPILER=LLVM_CLANG_BUILD_DIR/bin/clang \             
#   -DCLANG_REPOSITORY_STRING="https://llvm.googlesource.com/llvm-project" \
#   -DCMAKE_AR=LLVM_CLANG_BUILD_DIR/bin/llvm-ar \             
#   -DCMAKE_LINKER=LLVM_CLANG_BUILD_DIR/bin/ld.lld \             
#   -DCMAKE_NM=LLVM_CLANG_BUILD_DIR/bin/llvm-nm \             
#   -DCMAKE_OBJCOPY=LLVM_CLANG_BUILD_DIR/bin/llvm-objcopy \             
#   -DCMAKE_OBJDUMP=LLVM_CLANG_BUILD_DIR/bin/llvm-objdump \             
#   -DCMAKE_RANLIB=LLVM_CLANG_BUILD_DIR/bin/llvm-ranlib \             
#   -DCMAKE_READELF=LLVM_CLANG_BUILD_DIR/bin/llvm-readelf \             
#   -DCMAKE_STRIP=LLVM_CLANG_BUILD_DIR/bin/llvm-strip \             
#   -DCMAKE_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_aarch64-unknown-linux-gnu_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_armv7-unknown-linux-gnueabihf_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_i386-unknown-linux-gnu_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_riscv64-unknown-linux-gnu_SYSROOT=LINUX_SYSROOT_PATH \
#   -DLINUX_x86_64-unknown-linux-gnu_SYSROOT=LINUX_SYSROOT_PATH \
#   -DFUCHSIA_SDK=/mnt/nvme_crypt/SRC/llvm-prebuilts/fuchsia-idk \
#   -DLLVM_ENABLE_LTO=False \
#   -DLLVM_ENABLE_ASSERTIONS=True \
#   -DLLVM_ENABLE_BACKTRACES=True \                                            
#   -DFUCHSIA_TOOLCHAIN_RUNTIME_BUILD_MODE=RUNTIMES \                         
#   -DFUCHSIA_TOOLCHAIN_LINUX_RUNTIMES_TARGETS="x86_64-unknown-linux-gnu" \
#   -DLLVM_CMAKE_DIR=LLVM_CLANG_BUILD_DIR/lib/cmake/llvm \
#   -C ../runtimes/cmake/fuchsia/Linux.cmake ../runtimes

set(FUCHSIA_TOOLCHAIN_LINUX_RUNTIMES_TARGETS "" CACHE STRING "")

LIST(LENGTH FUCHSIA_TOOLCHAIN_LINUX_RUNTIMES_TARGETS target_count)
if (NOT FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_MODE STREQUAL "END2END")
  if (NOT target_count EQUAL 1)
    message(FATAL_ERROR "FUCHSIA_TOOLCHAIN_LINUX_RUNTIMES_TARGETS should only contain 1 target when not in END2END mode")
  endif()
  set(BUILTIN_TARGETS "" CACHE STRING "")
  set(RUNTIME_TARGETS "" CACHE STRING "")
else()
  if (target_count EQUAL 0)
    set(FUCHSIA_TOOLCHAIN_LINUX_RUNTIMES_TARGETS aarch64-unknown-linux-gnu;armv7-unknown-linux-gnueabihf;i386-unknown-linux-gnu;riscv64-unknown-linux-gnu;x86_64-unknown-linux-gnu)
  endif()
endif()

function(SET_LINUX_COMMON_ENTRIES RUNTIME_TARGET CONFIG_PREFIX)
  SET_COMMON_ENTRIES("${RUNTIME_TARGET}" "${CONFIG_PREFIX}" "Linux" "${LINUX_${target}_SYSROOT}" "--target=${target}" "-fuse-ld=lld")
endfunction()

function(SET_EXTRA_BUILTIN_ENTRIES RUNTIME_TARGET CONFIG_PREFIX)
  set(${CONFIG_PREFIX}COMPILER_RT_BUILD_STANDALONE_LIBATOMIC ON CACHE BOOL "")
  set(${CONFIG_PREFIX}COMPILER_RT_LIBATOMIC_USE_PTHREAD ON CACHE BOOL "")
endfunction()

function(SET_COMMON_STANDALONE_ARGS RUNTIME_TARGET)
  set(target ${RUNTIME_TARGET})
  if (NOT CMAKE_C_COMPILER)
    message(FATAL_ERROR "CMAKE_C_COMPILER not defined")
  endif()
  message(NOTICE "CMAKE_C_COMPILER is ${CMAKE_C_COMPILER}")
  cmake_path(GET CMAKE_C_COMPILER PARENT_PATH toolchain_bin)
  cmake_path(GET toolchain_bin PARENT_PATH toolchain_dir)
  set(CMAKE_C_COMPILER_WORKS ON CACHE STRING "")
  set(CMAKE_CXX_COMPILER_WORKS ON CACHE STRING "")
  set(CMAKE_ASM_COMPILER_WORKS ON CACHE STRING "")
  set(CMAKE_EXPORT_COMPILE_COMMANDS 1 CACHE STRING "")
  # TODO, set LLVM_HOST_TRIPLE LLVM_CONFIG_PATH
  set(PACKAGE_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}${LLVM_VERSION_SUFFIX}" CACHE STRING "")
  set(LLVM_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../llvm" CACHE STRING "")
  set(LLVM_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../llvm" CACHE STRING "")
  set(LLVM_BINARY_DIR ${toolchain_dir} CACHE STRING "")
  set(LLVM_HAVE_LINK_VERSION_SCRIPT 1 CACHE STRING "")
  set(LLVM_USE_RELATIVE_PATHS_IN_DEBUG_INFO OFF CACHE STRING "")
  set(LLVM_USE_RELATIVE_PATHS_IN_FILES ON CACHE STRING "")
  set(LLVM_ENABLE_WERROR OFF CACHE STRING "")
  set(LLVM_SOURCE_PREFIX "" CACHE STRING "")
  set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE STRING "")
  set(COMPILER_RT_DEFAULT_TARGET_ONLY ON CACHE STRING "")
  set(HAVE_LLVM_LIT ON CACHE STRING "")
  set(CLANG_RESOURCE_DIR "" CACHE STRING "")
  set(CMAKE_C_COMPILER_TARGET ${target} CACHE STRING "")
  set(CMAKE_CXX_COMPILER_TARGET ${target} CACHE STRING "")
  set(CMAKE_ASM_COMPILER_TARGET ${target} CACHE STRING "")
  set(LLVM_DEFAULT_TARGET_TRIPLE ${target} CACHE STRING "")
endfunction()

function(SET_BUILTINS_STANDALONE_ARGS RUNTIME_TARGET)
  cmake_path(GET CMAKE_C_COMPILER PARENT_PATH toolchain_bin)
  cmake_path(GET toolchain_bin PARENT_PATH toolchain_dir)
  set(LLVM_LIBRARY_OUTPUT_INTDIR "${toolchain_dir}/lib" CACHE STRING "")
  set(COMPILER_RT_INSTALL_BINARY_DIR "lib/clang/${LLVM_VERSION_MAJOR}/bin" CACHE STRING "")
  set(COMPILER_RT_INSTALL_DATA_DIR "lib/clang/${LLVM_VERSION_MAJOR}/share" CACHE STRING "")
  set(COMPILER_RT_INSTALL_INCLUDE_DIR "lib/clang/${LLVM_VERSION_MAJOR}/include" CACHE STRING "")
  set(COMPILER_RT_INSTALL_LIBRARY_DIR "lib/clang/${LLVM_VERSION_MAJOR}/lib" CACHE STRING "")
  set(COMPILER_RT_OUTPUT_DIR "${toolchain_dir}/lib/clang/${LLVM_VERSION_MAJOR}" CACHE STRING "")
endfunction()

function(SET_RUNTIMES_STANDALONE_ARGS RUNTIME_TARGET)
  set(target ${RUNTIME_TARGET})
  if (NOT CMAKE_C_COMPILER)
      message(FATAL_ERROR "CMAKE_C_COMPILER not defined")
  endif()
  cmake_path(GET CMAKE_C_COMPILER PARENT_PATH toolchain_bin)
  set(LLVM_ENABLE_PROJECTS_USED ON CACHE STRING "")
  set(LLVM_RUNTIMES_TARGET ${target} CACHE STRING "")
  set(COMPILER_RT_BUILD_BUILTINS OFF CACHE STRING "")
  set(LLVM_LIBC_FULL_BUILD ON CACHE STRING "")
  set(LLVM_USE_LINKER lld CACHE STRING "")
  set(LLVM_TOOLS_DIR ${toolchain_bin} CACHE STRING "")
  set(LLVM_INCLUDE_TESTS ON CACHE STRING "")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE STRING "")
endfunction()

function(SET_EXTRA_RUNTIMES_ENTRIES RUNTIME_TARGET CONFIG_PREFIX)
  set(target ${RUNTIME_TARGET})
  set(${CONFIG_PREFIX}COMPILER_RT_CXX_LIBRARY "libcxx" CACHE STRING "")
  set(${CONFIG_PREFIX}COMPILER_RT_USE_BUILTINS_LIBRARY ON CACHE BOOL "")
  set(${CONFIG_PREFIX}COMPILER_RT_USE_ATOMIC_LIBRARY ON CACHE BOOL "")
  set(${CONFIG_PREFIX}COMPILER_RT_USE_LLVM_UNWINDER ON CACHE BOOL "")
  set(${CONFIG_PREFIX}COMPILER_RT_CAN_EXECUTE_TESTS ON CACHE BOOL "")
  set(${CONFIG_PREFIX}COMPILER_RT_BUILD_STANDALONE_LIBATOMIC ON CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBUNWIND_ENABLE_SHARED OFF CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBUNWIND_USE_COMPILER_RT ON CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBCXXABI_USE_COMPILER_RT ON CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBCXXABI_ENABLE_SHARED OFF CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBCXXABI_USE_LLVM_UNWINDER ON CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBCXXABI_INSTALL_LIBRARY OFF CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBCXX_ENABLE_STATIC_ABI_LIBRARY ON CACHE BOOL "")
  set(${CONFIG_PREFIX}LIBCXX_ABI_VERSION 2 CACHE STRING "")
  set(${CONFIG_PREFIX}LLVM_ENABLE_ASSERTIONS OFF CACHE BOOL "")
  set(${CONFIG_PREFIX}SANITIZER_CXX_ABI "libc++" CACHE STRING "")
  set(${CONFIG_PREFIX}SANITIZER_CXX_ABI_INTREE ON CACHE BOOL "")
  set(${CONFIG_PREFIX}SANITIZER_TEST_CXX "libc++" CACHE STRING "")
  set(${CONFIG_PREFIX}SANITIZER_TEST_CXX_INTREE ON CACHE BOOL "")
  set(${CONFIG_PREFIX}LLVM_TOOLS_DIR "${CMAKE_BINARY_DIR}/bin" CACHE BOOL "")
  set(${CONFIG_PREFIX}LLVM_ENABLE_RUNTIMES "compiler-rt;libcxx;libcxxabi;libunwind" CACHE STRING "")
endfunction()


foreach(target IN LISTS FUCHSIA_TOOLCHAIN_LINUX_RUNTIMES_TARGETS)
  if (FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_MODE STREQUAL "END2END")
    SET_LINUX_COMMON_ENTRIES(${target} BUILTINS_${target}_)
    SET_EXTRA_BUILTIN_ENTRIES(${target} BUILTINS_${target}_)
    SET_LINUX_COMMON_ENTRIES(${target} RUNTIMES_${target}_)
    SET_EXTRA_RUNTIMES_ENTRIES(${target} RUNTIMES_${target}_)
    list(APPEND FUCHSIA_TOOLCHAIN_RUNTIME_BUILTIN_TARGETS "${target}")
    list(APPEND FUCHSIA_TOOLCHAIN_RUNTIME_RUNTIME_TARGETS "${target}")
    list(APPEND FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_ID_LINK "${target}")
  endif()
    
  if (FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_MODE STREQUAL "BUILTIN")
    SET_LINUX_COMMON_ENTRIES(${target} "")
    SET_EXTRA_BUILTIN_ENTRIES(${target} "")
    SET_COMMON_STANDALONE_ARGS(${target})
    SET_BUILTINS_STANDALONE_ARGS(${target})
  endif()

  if (FUCHSIA_TOOLCHAIN_RUNTIME_BUILD_MODE STREQUAL "RUNTIMES")
    SET_LINUX_COMMON_ENTRIES(${target} "")
    SET_EXTRA_RUNTIMES_ENTRIES(${target} "")
    SET_COMMON_STANDALONE_ARGS(${target})
    SET_RUNTIMES_STANDALONE_ARGS(${target})
  endif()
endforeach()
