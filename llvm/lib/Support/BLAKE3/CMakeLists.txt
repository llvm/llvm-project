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
  elseif(WIN32 OR CYGWIN)
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
else()
  # In a macOS Universal build (setting CMAKE_OSX_ARCHITECTURES to multiple
  # values), IS_X64 and IS_ARM64 won't be set, but compilation of the source
  # files will consider targeting either of them (each source file is
  # internally compiled once for each architecture). Thus, if we on the CMake
  # level decide not to include the assembly files, tell the source to not
  # expect it to be present either.
  #
  # Also, if targeting i386, then the blake3 source code automatically enables
  # the SIMD implementations, but we don't provide those sources.
  #
  # FIXME: We could improve the CMAKE_OSX_ARCHITECTURES configuration by
  # including all SIMD implementation files that might be relevant, and
  # wrapping them in ifdefs like "#ifdef __x86_64__", to allow them to be
  # included in a build for any architecture.
  add_definitions(-DBLAKE3_NO_AVX512 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_SSE2)
endif()

if (IS_ARM64)
  list(APPEND LLVM_BLAKE3_FILES
    blake3_neon.c
  )
else()
  add_definitions(-DBLAKE3_USE_NEON=0)
endif()

add_library(LLVMSupportBlake3 OBJECT EXCLUDE_FROM_ALL ${LLVM_BLAKE3_FILES})
llvm_update_compile_flags(LLVMSupportBlake3)
