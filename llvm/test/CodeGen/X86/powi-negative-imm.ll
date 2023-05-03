; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s

define void @test_powi(ptr %p) nounwind {
; CHECK-LABEL: powi:
; CHECK:             pushq %rax
; CHECK-NEXT:        movss {{.*#+}} xmm1 = mem[0],zero,zero,zero
; CHECK-COUNT-31:    mulss %xmm1, %xmm1
; CHECK-NEXT:        movss {{.*#+}} xmm0 = mem[0],zero,zero,zero
; CHECK-NEXT:        divss %xmm1, %xmm0
; CHECK-NEXT:        callq foo@PLT
; CHECK-NEXT:        popq %rax
; CHECK-NEXT:        retq
bb:
  %load = load float, ptr %p, align 4
  %call1 = call contract float @llvm.powi.f32.i32(float %load, i32 -2147483648)
  %call2 = call i1 @foo(float %call1)
  ret void
}

declare zeroext i1 @foo(float)
declare float @llvm.powi.f32.i32(float, i32)
