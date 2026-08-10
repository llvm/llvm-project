; RUN: llc < %s -mtriple=x86_64-apple-macosx10.10.0 -mattr=+sse2 | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.10.0 -mattr=+avx | FileCheck %s --check-prefixes=CHECK,AVX

; Assertions have been enhanced from utils/update_llc_test_checks.py to show the constant pool values.
; Use a macosx triple to make sure the format of those constant strings is exact.

; CHECK:       [[SIGNMASK1:L.+]]:
; CHECK-NEXT:  .long 0x80000000
; CHECK-NEXT:  .long 0x80000000
; CHECK-NEXT:  .long 0x80000000
; CHECK-NEXT:  .long 0x80000000

; CHECK:       [[MAGMASK1:L.+]]:
; CHECK-NEXT:  .long 0x7fffffff
; CHECK-NEXT:  .long 0x7fffffff
; CHECK-NEXT:  .long 0x7fffffff
; CHECK-NEXT:  .long 0x7fffffff

define <4 x float> @v4f32(<4 x float> %a, <4 x float> %b) nounwind {
; SSE2-LABEL: v4f32:
; SSE2:       # %bb.0:
; SSE2-NEXT:    andps [[SIGNMASK1]](%rip), %xmm1
; SSE2-NEXT:    andps [[MAGMASK1]](%rip), %xmm0
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: v4f32:
; AVX:       # %bb.0:
; AVX-NEXT:    vandps [[SIGNMASK1]](%rip), %xmm1, %xmm1
; AVX-NEXT:    vandps [[MAGMASK1]](%rip), %xmm0, %xmm0
; AVX-NEXT:    vorps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
;
  %tmp = tail call <4 x float> @llvm.copysign.v4f32( <4 x float> %a, <4 x float> %b )
  ret <4 x float> %tmp
}

; SSE2:       [[MAGMASK2:L.+]]:
; SSE2-NEXT:  .long 0x7fffffff
; SSE2-NEXT:  .long 0x7fffffff
; SSE2-NEXT:  .long 0x7fffffff
; SSE2-NEXT:  .long 0x7fffffff

; AVX:       [[SIGNMASK2:L.+]]:
; AVX-NEXT:  .long 0x80000000
; AVX-NEXT:  .long 0x80000000
; AVX-NEXT:  .long 0x80000000
; AVX-NEXT:  .long 0x80000000
; AVX-NEXT:  .long 0x80000000
; AVX-NEXT:  .long 0x80000000
; AVX-NEXT:  .long 0x80000000
; AVX-NEXT:  .long 0x80000000

; AVX:       [[MAGMASK2:L.+]]:
; AVX-NEXT:  .long 0x7fffffff
; AVX-NEXT:  .long 0x7fffffff
; AVX-NEXT:  .long 0x7fffffff
; AVX-NEXT:  .long 0x7fffffff
; AVX-NEXT:  .long 0x7fffffff
; AVX-NEXT:  .long 0x7fffffff
; AVX-NEXT:  .long 0x7fffffff
; AVX-NEXT:  .long 0x7fffffff

define <8 x float> @v8f32(<8 x float> %a, <8 x float> %b) nounwind {
; SSE2-LABEL: v8f32:
; SSE2:       # %bb.0:
; SSE2-NEXT:    movaps [[MAGMASK2]](%rip), %xmm4
; SSE2-NEXT:    movaps %xmm4, %xmm5
; SSE2-NEXT:    andnps %xmm2, %xmm5
; SSE2-NEXT:    andps %xmm4, %xmm0
; SSE2-NEXT:    orps %xmm5, %xmm0
; SSE2-NEXT:    andps %xmm4, %xmm1
; SSE2-NEXT:    andnps %xmm3, %xmm4
; SSE2-NEXT:    orps %xmm4, %xmm1
; SSE2-NEXT:    retq
;
; AVX-LABEL: v8f32:
; AVX:       # %bb.0:
; AVX-NEXT:    vandps [[SIGNMASK2]](%rip), %ymm1, %ymm1
; AVX-NEXT:    vandps [[MAGMASK2]](%rip), %ymm0, %ymm0
; AVX-NEXT:    vorps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
;
  %tmp = tail call <8 x float> @llvm.copysign.v8f32( <8 x float> %a, <8 x float> %b )
  ret <8 x float> %tmp
}

; CHECK:        [[SIGNMASK3:L.+]]:
; CHECK-NEXT:   .quad 0x8000000000000000
; CHECK-NEXT:   .quad 0x8000000000000000

; CHECK:        [[MAGMASK3:L.+]]:
; CHECK-NEXT:   .quad 0x7fffffffffffffff
; CHECK-NEXT:   .quad 0x7fffffffffffffff

define <2 x double> @v2f64(<2 x double> %a, <2 x double> %b) nounwind {
; SSE2-LABEL: v2f64:
; SSE2:       # %bb.0:
; SSE2-NEXT:    andps [[SIGNMASK3]](%rip), %xmm1
; SSE2-NEXT:    andps [[MAGMASK3]](%rip), %xmm0
; SSE2-NEXT:    orps %xmm1, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: v2f64:
; AVX:       # %bb.0:
; AVX-NEXT:    vandps [[SIGNMASK3]](%rip), %xmm1, %xmm1
; AVX-NEXT:    vandps [[MAGMASK3]](%rip), %xmm0, %xmm0
; AVX-NEXT:    vorps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
;
  %tmp = tail call <2 x double> @llvm.copysign.v2f64( <2 x double> %a, <2 x double> %b )
  ret <2 x double> %tmp
}

; SSE2:        [[MAGMASK4:L.+]]:
; SSE2-NEXT:   .quad 0x7fffffffffffffff
; SSE2-NEXT:   .quad 0x7fffffffffffffff

; AVX:        [[SIGNMASK4:L.+]]:
; AVX-NEXT:   .quad 0x8000000000000000
; AVX-NEXT:   .quad 0x8000000000000000
; AVX-NEXT:   .quad 0x8000000000000000
; AVX-NEXT:   .quad 0x8000000000000000

; AVX:        [[MAGMASK4:L.+]]:
; AVX-NEXT:   .quad 0x7fffffffffffffff
; AVX-NEXT:   .quad 0x7fffffffffffffff
; AVX-NEXT:   .quad 0x7fffffffffffffff
; AVX-NEXT:   .quad 0x7fffffffffffffff

define <4 x double> @v4f64(<4 x double> %a, <4 x double> %b) nounwind {
; SSE2-LABEL: v4f64:
; SSE2:       # %bb.0:
; SSE2-NEXT:    movaps [[MAGMASK4]](%rip), %xmm4
; SSE2-NEXT:    movaps %xmm4, %xmm5
; SSE2-NEXT:    andnps %xmm2, %xmm5
; SSE2-NEXT:    andps %xmm4, %xmm0
; SSE2-NEXT:    orps %xmm5, %xmm0
; SSE2-NEXT:    andps %xmm4, %xmm1
; SSE2-NEXT:    andnps %xmm3, %xmm4
; SSE2-NEXT:    orps %xmm4, %xmm1
; SSE2-NEXT:    retq
;
; AVX-LABEL: v4f64:
; AVX:       # %bb.0:
; AVX-NEXT:    vandps [[SIGNMASK4]](%rip), %ymm1, %ymm1
; AVX-NEXT:    vandps [[MAGMASK4]](%rip), %ymm0, %ymm0
; AVX-NEXT:    vorps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
;
  %tmp = tail call <4 x double> @llvm.copysign.v4f64( <4 x double> %a, <4 x double> %b )
  ret <4 x double> %tmp
}

declare <4 x float>     @llvm.copysign.v4f32(<4 x float>  %Mag, <4 x float>  %Sgn)
declare <8 x float>     @llvm.copysign.v8f32(<8 x float>  %Mag, <8 x float>  %Sgn)
declare <2 x double>    @llvm.copysign.v2f64(<2 x double> %Mag, <2 x double> %Sgn)
declare <4 x double>    @llvm.copysign.v4f64(<4 x double> %Mag, <4 x double> %Sgn)

