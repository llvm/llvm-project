set(LLVM_BLAKE3_FILES
  blake3.c
  blake3_dispatch.c
  blake3_portable.c
  blake3_neon.c
)

if (LLVM_DISABLE_ASSEMBLY_FILES)
  set(CAN_USE_ASSEMBLER FALSE)
else()
  set(CAN_USE_ASSEMBLER TRUE)
endif()

macro(disable_blake3_x86_simd)
  add_definitions(-DBLAKE3_NO_AVX512 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_SSE2)
endmacro()

# The BLAKE3 team recommends using the assembly versions, from the README:
#
# "For each of the x86 SIMD instruction sets, four versions are available:
# three flavors of assembly (Unix, Windows MSVC, and Windows GNU) and one
# version using C intrinsics. The assembly versions are generally
# preferred. They perform better, they perform more consistently across
# different compilers, and they build more quickly."

if (CAN_USE_ASSEMBLER)
  if (MSVC)
    check_symbol_exists(_M_X64 "" IS_X64)
    if (IS_X64)
      enable_language(ASM_MASM)
      list(APPEND LLVM_BLAKE3_FILES
        blake3_sse2_x86-64_windows_msvc.asm
        blake3_sse41_x86-64_windows_msvc.asm
        blake3_avx2_x86-64_windows_msvc.asm
        blake3_avx512_x86-64_windows_msvc.asm
      )
    else()
      disable_blake3_x86_simd()
    endif()
  elseif(WIN32 OR CYGWIN)
    check_symbol_exists(__x86_64__ "" IS_X64)
    if (IS_X64)
      list(APPEND LLVM_BLAKE3_FILES
        blake3_sse2_x86-64_windows_gnu.S
        blake3_sse41_x86-64_windows_gnu.S
        blake3_avx2_x86-64_windows_gnu.S
        blake3_avx512_x86-64_windows_gnu.S
      )
      # Clang-6 needs this flag.
      set_source_files_properties(blake3_avx512_x86-64_windows_gnu.S
        PROPERTIES COMPILE_OPTIONS "-mavx512vl")
    else()
      disable_blake3_x86_simd()
    endif()
  else()
    check_symbol_exists(__i386__ "" IS_X32)
    if (IS_X32)
      # blake3 C code autoenables SIMD for i386, but those implementations are
      # only available via the intrinsics sources which we don't enable here.
      disable_blake3_x86_simd()
    else()
      # In a macOS Universal build (setting CMAKE_OSX_ARCHITECTURES to multiple
      # values), compilation of the source files will target multiple architectures
      # (each source file is internally compiled once for each architecture).
      # To accomodate this configuration we include these assembly files without a
      # CMake check but their source is guarded with architecture "#ifdef" checks.
      list(APPEND LLVM_BLAKE3_FILES
        blake3_sse2_x86-64_unix.S
        blake3_sse41_x86-64_unix.S
        blake3_avx2_x86-64_unix.S
        blake3_avx512_x86-64_unix.S
      )
      check_c_compiler_flag("-mavx512vl" SUPPORTS_avx512vl)
      if (SUPPORTS_avx512vl)
        # Clang-6 needs this flag. We also suppress '-Wunused-command-line-argument'
        # in case the file is included with arm architecture.
        set_source_files_properties(blake3_avx512_x86-64_unix.S
          PROPERTIES COMPILE_OPTIONS "-mavx512vl;-Wno-unused-command-line-argument")
      endif()
    endif()
  endif()
else()
  # CAN_USE_ASSEMBLER == FALSE
  disable_blake3_x86_simd()
endif()

add_library(LLVMSupportBlake3 OBJECT EXCLUDE_FROM_ALL ${LLVM_BLAKE3_FILES})
llvm_update_compile_flags(LLVMSupportBlake3)
