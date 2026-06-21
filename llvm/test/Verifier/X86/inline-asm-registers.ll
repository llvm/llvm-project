; The X86 target verifier flags inline asm that names a physical register only
; available on a subtarget with a feature the selected subtarget lacks.
;
; RUN: opt -passes=target-verifier -disable-output %s 2>&1 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; Clobbering a zmm register requires AVX-512 (default x86_64 is SSE2-level).
; CHECK: inline asm references an AVX-512 register (zmm/k), but the subtarget does not support AVX-512.
define void @asm_zmm_without_avx512() {
  call void asm sideeffect "", "~{zmm16}"()
  ret void
}

; Clobbering a mask register (k1) likewise requires AVX-512.
; CHECK: inline asm references an AVX-512 register (zmm/k), but the subtarget does not support AVX-512.
define void @asm_mask_without_avx512() {
  call void asm sideeffect "", "~{k1}"()
  ret void
}

; Clobbering a ymm register requires AVX.
; CHECK: inline asm references an AVX register (ymm), but the subtarget does not support AVX.
define void @asm_ymm_without_avx() {
  call void asm sideeffect "", "~{ymm0}"()
  ret void
}
