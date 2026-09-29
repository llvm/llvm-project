; RUN: llc -mtriple=i686-unknown-unknown -mattr=+sse4.1 < %s | FileCheck %s -check-prefix=X32
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 < %s | FileCheck %s -check-prefix=X64

; Test for case where insertps was folding the load of the insertion element, but a later optimization
; was then manipulating the load.

define <4 x float> @insertps_unfold(ptr %v0, ptr %v1) {
; X32-LABEL: insertps_unfold:
; X32:       # %bb.0:
; X32-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X32-NEXT:    movl {{[0-9]+}}(%esp), %ecx
; X32-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X32-NEXT:    movaps (%eax), %xmm0
; X32-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],mem[0]
; X32-NEXT:    addps %xmm1, %xmm0
; X32-NEXT:    retl
;
; X64-LABEL: insertps_unfold:
; X64:       # %bb.0:
; X64-NEXT:    movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; X64-NEXT:    movaps (%rdi), %xmm0
; X64-NEXT:    insertps {{.*#+}} xmm0 = xmm0[0,1,2],mem[0]
; X64-NEXT:    addps %xmm1, %xmm0
; X64-NEXT:    retq
  %a = getelementptr inbounds <4 x float>, ptr %v1, i64 0, i64 1
  %b = load float, ptr %a, align 4
  %c = insertelement <4 x float> undef, float %b, i32 0
  %d = load <4 x float>, ptr %v1, align 16
  %e = load <4 x float>, ptr %v0, align 16
  %f = shufflevector <4 x float> %e, <4 x float> %d, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  %g = fadd <4 x float> %c, %f
  ret <4 x float> %g
}
