set(LLVM_BLAKE3_FILES
  blake3.c
  blake3_dispatch.c
  blake3_portable.c
)

if (LLVM_DISABLE_ASSEMBLY_FILES)
  set(CAN_USE_ASSEMBLER FALSE)
else()
  set(CAN_USE_ASSEMBLER TRUE)
endif()

# Some LLVM builders set "CC=ccache /path/to/clang" as environment variable,
# which confuses CMake thinking the assembler to use is the 'ccache' binary.
# These builders need to also set "ASM=/path/to/clang" in environment, or use
# LLVM_CCACHE_BUILD CMake variable instead, but until this happens check for
# this case and disable building the assembly files.
get_filename_component(ASM_BINARY ${CMAKE_ASM_COMPILER} NAME)
if (ASM_BINARY STREQUAL "ccache")
  message(WARNING "CMake is set to use '${CMAKE_ASM_COMPILER}' as assembler "
    "which is likely due to having 'CC=ccache /path/to/clang' in the environment. "
    "Building the BLAKE3 SIMD-optimized assembly files is disabled, set "
    "'ASM=/path/to/clang' as environment variable and do a clean re-configure, "
    "or unset CC/CXX and configure with '-DLLVM_CCACHE_BUILD=YES' instead, "
    "to correct this.")
  set(CAN_USE_ASSEMBLER FALSE)
endif()

# The BLAKE3 team recommends using the assembly versions, from the README:
#
# "For each of the x86 SIMD instruction sets, four versions are available:
# three flavors of assembly (Unix, Windows MSVC, and Windows GNU) and one
# version using C intrinsics. The assembly versions are generally
# preferred. They perform better, they perform more consistently across
# different compilers, and they build more quickly."

if (MSVC)
  check_symbol_exists(_M_X64 "" IS_X64)
  check_symbol_exists(_M_ARM64 "" IS_ARM64)
else()
  check_symbol_exists(__x86_64__ "" IS_X64)
  check_symbol_exists(__aarch64__ "" IS_ARM64)

  if (IS_X64)
    # Clang-6 needs this flag.
    set_source_files_properties(blake3_avx512_x86-64_windows_gnu.S
      PROPERTIES COMPILE_OPTIONS "-mavx512vl")
    set_source_files_properties(blake3_avx512_x86-64_unix.S
      PROPERTIES COMPILE_OPTIONS "-mavx512vl")
  endif()
endif()

if (IS_X64 AND CAN_USE_ASSEMBLER)
  if (MSVC)
    enable_language(ASM_MASM)
    list(APPEND LLVM_BLAKE3_FILES
      blake3_sse2_x86-64_windows_msvc.asm
      blake3_sse41_x86-64_windows_msvc.asm
      blake3_avx2_x86-64_windows_msvc.asm
      blake3_avx512_x86-64_windows_msvc.asm
    )
  elseif(WIN32)
    list(APPEND LLVM_BLAKE3_FILES
      blake3_sse2_x86-64_windows_gnu.S
      blake3_sse41_x86-64_windows_gnu.S
      blake3_avx2_x86-64_windows_gnu.S
      blake3_avx512_x86-64_windows_gnu.S
    )
  else()
    list(APPEND LLVM_BLAKE3_FILES
      blake3_sse2_x86-64_unix.S
      blake3_sse41_x86-64_unix.S
      blake3_avx2_x86-64_unix.S
      blake3_avx512_x86-64_unix.S
    )
  endif()
endif()

if (IS_ARM64)
  list(APPEND LLVM_BLAKE3_FILES
    blake3_neon.c
  )
endif()

if (IS_X64 AND NOT CAN_USE_ASSEMBLER)
  add_definitions(-DBLAKE3_NO_AVX512 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_SSE2)
endif()

add_library(LLVMSupportBlake3 OBJECT EXCLUDE_FROM_ALL ${LLVM_BLAKE3_FILES})
llvm_update_compile_flags(LLVMSupportBlake3)
