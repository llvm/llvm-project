cmake_minimum_required(VERSION 3.13.4)

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../libunwind")
  message(FATAL_ERROR "llvm-libgcc requires being built in a monorepo layout with libunwind available")
endif()

set(LLVM_COMMON_CMAKE_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  "${LLVM_COMMON_CMAKE_UTILS}"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

project(llvm-libgcc LANGUAGES C CXX ASM)

if(NOT LLVM_LIBGCC_EXPLICIT_OPT_IN)
  message(FATAL_ERROR
    "llvm-libgcc is not for the casual LLVM user. It is intended to be used by distro "
    "managers who want to replace libgcc with compiler-rt and libunwind, but cannot "
    "fully abandon the libgcc family (e.g. because they are dependent on glibc). Such "
    "managers must have worked out their compatibility requirements ahead of using "
    "llvm-libgcc. If you want to build llvm-libgcc, please add -DLLVM_LIBGCC_EXPLICIT_OPT_IN=Yes "
    "to your CMake invocation and try again.")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib${LLVMLIB_DIR_SUFFIX}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib${LLVMLIB_DIR_SUFFIX}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

set(COMPILER_RT_BUILD_BUILTINS On)
set(COMPILER_RT_BUILTINS_HIDE_SYMBOLS Off)
add_subdirectory("../compiler-rt" "compiler-rt")

set(LIBUNWIND_ENABLE_STATIC On)
set(LIBUNWIND_ENABLE_SHARED Off)
set(LIBUNWIND_HAS_COMMENT_LIB_PRAGMA Off)
set(LIBUNWIND_USE_COMPILER_RT On)
add_subdirectory("../libunwind" "libunwind")

add_subdirectory(lib)
