set(LLVM_BLAKE3_FILES
  blake3.c
  blake3_dispatch.c
  blake3_portable.c
)

# The BLAKE3 team recommends using the assembly versions, from the README:
#
# "For each of the x86 SIMD instruction sets, four versions are available:
# three flavors of assembly (Unix, Windows MSVC, and Windows GNU) and one
# version using C intrinsics. The assembly versions are generally
# preferred. They perform better, they perform more consistently across
# different compilers, and they build more quickly."
# FIXME: Figure out what is wrong with the builders when using the assembly files and neon.
if (MSVC)
  enable_language(ASM_MASM)
endif()

if (FALSE)#CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  if (MSVC)
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

if (FALSE)#CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch)")
  list(APPEND LLVM_BLAKE3_FILES
    blake3_neon.c
  )
endif()

# FIXME: Figure out what is wrong with the builders when using the assembly files.
add_definitions(-DBLAKE3_NO_AVX512 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_SSE2 -DBLAKE3_USE_NEON=0)

add_library(LLVMSupportBlake3 OBJECT EXCLUDE_FROM_ALL ${LLVM_BLAKE3_FILES})
llvm_update_compile_flags(LLVMSupportBlake3)
