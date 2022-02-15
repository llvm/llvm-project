; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

declare <4 x i32> @llvm.vp.fptosi.v4i32.v8f32(<8 x float>, <4 x i1>, i32);

; CHECK: VP cast intrinsic first argument and result vector lengths must be equal
; CHECK-NEXT: %r0 = call <4 x i32>

define void @test_vp_fptosi(<8 x float> %src, <4 x i1> %m, i32 %n) {
  %r0 = call <4 x i32> @llvm.vp.fptosi.v4i32.v8f32(<8 x float> %src, <4 x i1> %m, i32 %n)
  ret void
}
