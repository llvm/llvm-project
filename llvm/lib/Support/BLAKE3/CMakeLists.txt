set(LLVM_BLAKE3_FILES
  BLAKE3/blake3.c
  BLAKE3/blake3_dispatch.c
  BLAKE3/blake3_portable.c
)

# The BLAKE3 team recommends using the assembly versions, from the README:
#
# "For each of the x86 SIMD instruction sets, four versions are available:
# three flavors of assembly (Unix, Windows MSVC, and Windows GNU) and one
# version using C intrinsics. The assembly versions are generally
# preferred. They perform better, they perform more consistently across
# different compilers, and they build more quickly."
# FIXME: Figure out what is wrong with the builders when using the assembly files and neon.
if (FALSE)#CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  if (MSVC)
    list(APPEND LLVM_BLAKE3_FILES
      BLAKE3/blake3_sse2_x86-64_windows_msvc.asm
      BLAKE3/blake3_sse41_x86-64_windows_msvc.asm
      BLAKE3/blake3_avx2_x86-64_windows_msvc.asm
      BLAKE3/blake3_avx512_x86-64_windows_msvc.asm
    )
  elseif(WIN32)
    list(APPEND LLVM_BLAKE3_FILES
      BLAKE3/blake3_sse2_x86-64_windows_gnu.S
      BLAKE3/blake3_sse41_x86-64_windows_gnu.S
      BLAKE3/blake3_avx2_x86-64_windows_gnu.S
      BLAKE3/blake3_avx512_x86-64_windows_gnu.S
    )
  else()
    list(APPEND LLVM_BLAKE3_FILES
      BLAKE3/blake3_sse2_x86-64_unix.S
      BLAKE3/blake3_sse41_x86-64_unix.S
      BLAKE3/blake3_avx2_x86-64_unix.S
      BLAKE3/blake3_avx512_x86-64_unix.S
    )
  endif()
endif()

if (FALSE)#CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch)")
  list(APPEND LLVM_BLAKE3_FILES
    BLAKE3/blake3_neon.c
  )
endif()

set(LLVM_BLAKE3_FILES
  ${LLVM_BLAKE3_FILES}
  PARENT_SCOPE
)
