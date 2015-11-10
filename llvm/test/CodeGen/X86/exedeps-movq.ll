; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=sse2 | FileCheck %s --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx | FileCheck %s --check-prefix=AVX

; Verify that we select the correct version of the instruction that stores the low 64-bits
; of a 128-bit vector. We want to avoid int/fp domain crossing penalties, so ignore the
; bitcast ops and choose:
;
; movlps for floats
; movlpd for doubles
; movq for integers

define void @store_floats(<4 x float> %x, i64* %p) {
; SSE-LABEL: store_floats:
; SSE:       # BB#0:
; SSE-NEXT:    addps %xmm0, %xmm0
; SSE-NEXT:    movlps %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX-LABEL: store_floats:
; AVX:       # BB#0:
; AVX-NEXT:    vaddps %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vmovlps %xmm0, (%rdi)
; AVX-NEXT:    retq
  %a = fadd <4 x float> %x, %x
  %b = shufflevector <4 x float> %a, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  %c = bitcast <2 x float> %b to i64
  store i64 %c, i64* %p
  ret void
}

define void @store_double(<2 x double> %x, i64* %p) {
; SSE-LABEL: store_double:
; SSE:       # BB#0:
; SSE-NEXT:    addpd %xmm0, %xmm0
; SSE-NEXT:    movlpd %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX-LABEL: store_double:
; AVX:       # BB#0:
; AVX-NEXT:    vaddpd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vmovlpd %xmm0, (%rdi)
; AVX-NEXT:    retq
  %a = fadd <2 x double> %x, %x
  %b = extractelement <2 x double> %a, i32 0
  %c = bitcast double %b to i64
  store i64 %c, i64* %p
  ret void
}

define void @store_int(<4 x i32> %x, <2 x float>* %p) {
; SSE-LABEL: store_int:
; SSE:       # BB#0:
; SSE-NEXT:    paddd %xmm0, %xmm0
; SSE-NEXT:    movq %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX-LABEL: store_int:
; AVX:       # BB#0:
; AVX-NEXT:    vpaddd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vmovq %xmm0, (%rdi)
; AVX-NEXT:    retq
  %a = add <4 x i32> %x, %x
  %b = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
  %c = bitcast <2 x i32> %b to <2 x float>
  store <2 x float> %c, <2 x float>* %p
  ret void
}

define void @store_h_double(<2 x double> %x, i64* %p) {
; SSE-LABEL: store_h_double:
; SSE:       # BB#0:
; SSE-NEXT:    addpd %xmm0, %xmm0
; SSE-NEXT:    movhpd %xmm0, (%rdi)
; SSE-NEXT:    retq
;
; AVX-LABEL: store_h_double:
; AVX:       # BB#0:
; AVX-NEXT:    vaddpd %xmm0, %xmm0, %xmm0
; AVX-NEXT:    vmovhpd %xmm0, (%rdi)
; AVX-NEXT:    retq
  %a = fadd <2 x double> %x, %x
  %b = extractelement <2 x double> %a, i32 1
  %c = bitcast double %b to i64
  store i64 %c, i64* %p
  ret void
}
