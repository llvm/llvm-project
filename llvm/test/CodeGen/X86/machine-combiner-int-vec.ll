; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=sse2 -machine-combiner-verify-pattern-order=true < %s | FileCheck %s --check-prefix=SSE
; RUN: llc -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -mattr=avx2 -machine-combiner-verify-pattern-order=true < %s | FileCheck %s --check-prefix=AVX

; Verify that 128-bit vector logical ops are reassociated.

define <4 x i32> @reassociate_and_v4i32(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, <4 x i32> %x3) {
; SSE-LABEL: reassociate_and_v4i32:
; SSE:       # %bb.0:
; SSE-NEXT:    paddd %xmm1, %xmm0
; SSE-NEXT:    pand %xmm3, %xmm2
; SSE-NEXT:    pand %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_and_v4i32:
; AVX:       # %bb.0:
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq

  %t0 = add <4 x i32> %x0, %x1
  %t1 = and <4 x i32> %x2, %t0
  %t2 = and <4 x i32> %x3, %t1
  ret <4 x i32> %t2
}

define <4 x i32> @reassociate_or_v4i32(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, <4 x i32> %x3) {
; SSE-LABEL: reassociate_or_v4i32:
; SSE:       # %bb.0:
; SSE-NEXT:    paddd %xmm1, %xmm0
; SSE-NEXT:    por %xmm3, %xmm2
; SSE-NEXT:    por %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_or_v4i32:
; AVX:       # %bb.0:
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpor %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vpor %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq

  %t0 = add <4 x i32> %x0, %x1
  %t1 = or <4 x i32> %x2, %t0
  %t2 = or <4 x i32> %x3, %t1
  ret <4 x i32> %t2
}

define <4 x i32> @reassociate_xor_v4i32(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, <4 x i32> %x3) {
; SSE-LABEL: reassociate_xor_v4i32:
; SSE:       # %bb.0:
; SSE-NEXT:    paddd %xmm1, %xmm0
; SSE-NEXT:    pxor %xmm3, %xmm2
; SSE-NEXT:    pxor %xmm2, %xmm0
; SSE-NEXT:    retq
;
; AVX-LABEL: reassociate_xor_v4i32:
; AVX:       # %bb.0:
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpxor %xmm3, %xmm2, %xmm1
; AVX-NEXT:    vpxor %xmm1, %xmm0, %xmm0
; AVX-NEXT:    retq

  %t0 = add <4 x i32> %x0, %x1
  %t1 = xor <4 x i32> %x2, %t0
  %t2 = xor <4 x i32> %x3, %t1
  ret <4 x i32> %t2
}

; Verify that 256-bit vector logical ops are reassociated.

define <8 x i32> @reassociate_and_v8i32(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, <8 x i32> %x3) {
; AVX-LABEL: reassociate_and_v8i32:
; AVX:       # %bb.0:
; AVX-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vpand %ymm3, %ymm2, %ymm1
; AVX-NEXT:    vpand %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq

  %t0 = add <8 x i32> %x0, %x1
  %t1 = and <8 x i32> %x2, %t0
  %t2 = and <8 x i32> %x3, %t1
  ret <8 x i32> %t2
}

define <8 x i32> @reassociate_or_v8i32(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, <8 x i32> %x3) {
; AVX-LABEL: reassociate_or_v8i32:
; AVX:       # %bb.0:
; AVX-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vpor %ymm3, %ymm2, %ymm1
; AVX-NEXT:    vpor %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq

  %t0 = add <8 x i32> %x0, %x1
  %t1 = or <8 x i32> %x2, %t0
  %t2 = or <8 x i32> %x3, %t1
  ret <8 x i32> %t2
}

define <8 x i32> @reassociate_xor_v8i32(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, <8 x i32> %x3) {
; AVX-LABEL: reassociate_xor_v8i32:
; AVX:       # %bb.0:
; AVX-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vpxor %ymm3, %ymm2, %ymm1
; AVX-NEXT:    vpxor %ymm1, %ymm0, %ymm0
; AVX-NEXT:    retq

  %t0 = add <8 x i32> %x0, %x1
  %t1 = xor <8 x i32> %x2, %t0
  %t2 = xor <8 x i32> %x3, %t1
  ret <8 x i32> %t2
}

