cmake_minimum_required(VERSION 3.4.3)

set(ENABLE_EXPERIMENTAL_NEW_PASS_MANAGER ON CACHE BOOL "")
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "")
set(LLVM_ENABLE_CLASSIC_FLANG ON CACHE BOOL "")
set(LLVM_TARGETS_TO_BUILD "X86;Next32;RISCV" CACHE STRING "")
set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
set(LLVM_INSTALL_BINUTILS_SYMLINKS ON CACHE BOOL "")
set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
set(LLVM_ENABLE_PLUGINS OFF CACHE BOOL "")

set(CLANG_VENDOR "NextSilicon" CACHE STRING "")
set(PACKAGE_VENDOR "NextSilicon" CACHE STRING "")
set(BUG_REPORT_URL "Support@nextsilicon.com" CACHE STRING "")
set(CLANG_DEFAULT_RTLIB "compiler-rt" CACHE STRING "")
set(CLANG_DEFAULT_UNWINDLIB "libunwind" CACHE STRING "")
set(CLANG_DEFAULT_CXX_STDLIB "libc++" CACHE STRING "")


# minimize targets
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "We don't want ocaml bindings")
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "No need for libedit")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "No need for terminfo")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(CLANG_ENABLE_ARCMT OFF CACHE BOOL "")
set(CLANG_ENABLE_STATIC_ANALYZER OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_ENABLE_DIA_SDK OFF CACHE BOOL "Lets try to remove targets")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_ENABLE_OCAMLDOC OFF CACHE BOOL "")

set(CMAKE_BUILD_WITH_INSTALL_RPATH ON CACHE BOOL "Use install RPATH throughout the build")

#set(LLVM_ENABLE_LTO ON CACHE BOOL "")
#set(LLVM_USE_RELATIVE_PATHS_IN_FILES ON CACHE BOOL "")

set(LLD_SYMLINKS_TO_CREATE lld-link ld.lld ld64.lld)
set(LLVM_INSTALL_UTILS ON CACHE BOOL "Needed for tests")
set(LLVM_TOOLCHAIN_UTILITIES
  FileCheck
  not
  count
  llvm-headers
  cmake-exports
  llvm-libraries
  llvm-link
  llvm-reduce
  llvm-diff
  llc
  CACHE STRING "")

set(LLVM_DISTRIBUTION_COMPONENTS
  clang
  lld
  LTO
  clang-resource-headers
  clang-offload-packager
  clang-linker-wrapper
  llvm-config
  llvm-symbolizer
  addr2line
  llvm-addr2line
  llvm-ar
  ar
  llvm-ranlib
  ranlib
  llvm-objcopy
  objcopy
  llvm-objdump
  objdump
  llvm-dwarfdump
  readelf
  llvm-readelf
  llvm-readobj
  llvm-nm
  llvm-dis
  nm
  llvm-size
  size
  opt
  # headers and stuff for nextutils
  ${LLVM_TOOLCHAIN_UTILITIES}
  CACHE STRING "")

set(clang_links)
foreach(triple_prefix "" "next32-unknown-linux-" "riscv64-unknown-elf-")
    foreach(base "clang" "clang++" "clang-cl" "clang-cpp" "flang")
        list(APPEND clang_links "${triple_prefix}${base}")
    endforeach()
endforeach()
list(REMOVE_ITEM clang_links "clang")

set(CLANG_LINKS_TO_CREATE "${clang_links}" CACHE STRING "")

option(USE_CCACHE "Use ccache to speed up recompilation" ON)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND ${USE_CCACHE})
    set(LLVM_CCACHE_BUILD ON CACHE BOOL "")
endif()
