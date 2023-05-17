; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

define float @test_powi(ptr %p) nounwind {
; CHECK-LABEL: test_powi:
; CHECK:       # %bb.0: # %bb
; CHECK-NEXT:        movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; CHECK-COUNT-31:    mulss %xmm1, %xmm1
; CHECK-NEXT:        movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; CHECK-NEXT:        divss %xmm1, %xmm0
; CHECK-NEXT:        retq
bb:
  %load = load float, ptr %p, align 4
  %call = call contract float @llvm.powi.f32.i32(float %load, i32 -2147483648)
  ret float %call
}

declare float @llvm.powi.f32.i32(float, i32)
