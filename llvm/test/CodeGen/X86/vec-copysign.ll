; RUN: llc < %s -mtriple=x86_64-apple-macosx10.10.0 -mattr=+sse2 | FileCheck %s --check-prefixes=CHECK,SSE2
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.10.0 -mattr=+avx | FileCheck %s --check-prefixes=CHECK,AVX

; Assertions have been enhanced from utils/update_llc_test_checks.py to show the constant pool values.
; Use a macosx triple to make sure the format of those constant strings is exact.

; CHECK:       [[SIGNMASK1:L.+]]:
; CHECK-NEXT:  .long 0x80000000
; CHECK-NEXT:  .long 0x80000000
; CHECK-NEXT:  .long 0x80000000
; CHECK-NEXT:  .long 0x80000000

; AVX:       [[SIGNSPLAT1:L.+]]:
; AVX-NEXT:  .long 0x80000000

define <4 x float> @v4f32(<4 x float> %a, <4 x float> %b) nounwind {
; SSE2-LABEL: v4f32:
; SSE2:       ## %bb.0:
; SSE2-NEXT:    movaps [[SIGNMASK1]](%rip), %xmm2
; SSE2-NEXT:    andps %xmm2, %xmm1
; SSE2-NEXT:    andnps %xmm0, %xmm2
; SSE2-NEXT:    orps %xmm1, %xmm2
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: v4f32:
; AVX:       ## %bb.0:
; AVX-NEXT:    vbroadcastss [[SIGNSPLAT1]](%rip), %xmm2
; AVX-NEXT:    vandps %xmm2, %xmm1, %xmm1
; AVX-NEXT:    vandnps %xmm0, %xmm2, %xmm0
; AVX-NEXT:    vorps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %tmp = tail call <4 x float> @llvm.copysign.v4f32( <4 x float> %a, <4 x float> %b )
  ret <4 x float> %tmp
}

; CHECK:       [[SIGNMASK2:L.+]]:
; CHECK-NEXT:  .long 0x80000000
; CHECK-NEXT:  .long 0x80000000
; CHECK-NEXT:  .long 0x80000000
; CHECK-NEXT:  .long 0x80000000

; AVX:       [[SIGNSPLAT2:L.+]]:
; AVX-NEXT:  .long 0x80000000

define <8 x float> @v8f32(<8 x float> %a, <8 x float> %b) nounwind {
; SSE2-LABEL: v8f32:
; SSE2:       ## %bb.0:
; SSE2-NEXT:    movaps [[SIGNMASK2]](%rip), %xmm4
; SSE2-NEXT:    andps %xmm4, %xmm2
; SSE2-NEXT:    movaps %xmm4, %xmm5
; SSE2-NEXT:    andnps %xmm0, %xmm5
; SSE2-NEXT:    orps %xmm2, %xmm5
; SSE2-NEXT:    andps %xmm4, %xmm3
; SSE2-NEXT:    andnps %xmm1, %xmm4
; SSE2-NEXT:    orps %xmm3, %xmm4
; SSE2-NEXT:    movaps %xmm5, %xmm0
; SSE2-NEXT:    movaps %xmm4, %xmm1
; SSE2-NEXT:    retq
;
; AVX-LABEL: v8f32:
; AVX:       ## %bb.0:
; AVX-NEXT:    vbroadcastss [[SIGNSPLAT2]](%rip), %ymm2
; AVX-NEXT:    vandps %ymm2, %ymm1, %ymm1
; AVX-NEXT:    vandnps %ymm0, %ymm2, %ymm0
; AVX-NEXT:    vorps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %tmp = tail call <8 x float> @llvm.copysign.v8f32( <8 x float> %a, <8 x float> %b )
  ret <8 x float> %tmp
}

; CHECK:        [[SIGNMASK3:L.+]]:
; CHECK-NEXT:   .quad 0x8000000000000000
; CHECK-NEXT:   .quad 0x8000000000000000

; AVX:        [[SIGNSPLAT3:L.+]]:
; AVX-NEXT:   .quad 0x8000000000000000

define <2 x double> @v2f64(<2 x double> %a, <2 x double> %b) nounwind {
; SSE2-LABEL: v2f64:
; SSE2:       ## %bb.0:
; SSE2-NEXT:    movaps [[SIGNMASK3]](%rip), %xmm2
; SSE2-NEXT:    andps %xmm2, %xmm1
; SSE2-NEXT:    andnps %xmm0, %xmm2
; SSE2-NEXT:    orps %xmm1, %xmm2
; SSE2-NEXT:    movaps %xmm2, %xmm0
; SSE2-NEXT:    retq
;
; AVX-LABEL: v2f64:
; AVX:       ## %bb.0:
; AVX-NEXT:    vmovddup [[SIGNSPLAT3]](%rip), %xmm2
; AVX-NEXT:    ## xmm2 = mem[0,0]
; AVX-NEXT:    vandps %xmm2, %xmm1, %xmm1
; AVX-NEXT:    vandnps %xmm0, %xmm2, %xmm0
; AVX-NEXT:    vorps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq
  %tmp = tail call <2 x double> @llvm.copysign.v2f64( <2 x double> %a, <2 x double> %b )
  ret <2 x double> %tmp
}

; CHECK:        [[SIGNMASK4:L.+]]:
; CHECK-NEXT:   .quad 0x8000000000000000
; CHECK-NEXT:   .quad 0x8000000000000000

; AVX:        [[SIGNSPLAT4:L.+]]:
; AVX-NEXT:   .quad 0x8000000000000000

define <4 x double> @v4f64(<4 x double> %a, <4 x double> %b) nounwind {
; SSE2-LABEL: v4f64:
; SSE2:       ## %bb.0:
; SSE2-NEXT:    movaps [[SIGNMASK4]](%rip), %xmm4
; SSE2-NEXT:    andps %xmm4, %xmm2
; SSE2-NEXT:    movaps %xmm4, %xmm5
; SSE2-NEXT:    andnps %xmm0, %xmm5
; SSE2-NEXT:    orps %xmm2, %xmm5
; SSE2-NEXT:    andps %xmm4, %xmm3
; SSE2-NEXT:    andnps %xmm1, %xmm4
; SSE2-NEXT:    orps %xmm3, %xmm4
; SSE2-NEXT:    movaps %xmm5, %xmm0
; SSE2-NEXT:    movaps %xmm4, %xmm1
; SSE2-NEXT:    retq
;
; AVX-LABEL: v4f64:
; AVX:       ## %bb.0:
; AVX-NEXT:    vbroadcastsd [[SIGNSPLAT4]](%rip), %ymm2
; AVX-NEXT:    vandps %ymm2, %ymm1, %ymm1
; AVX-NEXT:    vandnps %ymm0, %ymm2, %ymm0
; AVX-NEXT:    vorps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq
  %tmp = tail call <4 x double> @llvm.copysign.v4f64( <4 x double> %a, <4 x double> %b )
  ret <4 x double> %tmp
}

declare <4 x float>     @llvm.copysign.v4f32(<4 x float>  %Mag, <4 x float>  %Sgn)
declare <8 x float>     @llvm.copysign.v8f32(<8 x float>  %Mag, <8 x float>  %Sgn)
declare <2 x double>    @llvm.copysign.v2f64(<2 x double> %Mag, <2 x double> %Sgn)
declare <4 x double>    @llvm.copysign.v4f64(<4 x double> %Mag, <4 x double> %Sgn)

