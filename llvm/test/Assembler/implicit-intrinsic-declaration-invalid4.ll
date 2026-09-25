; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Use of intrinsic with an invalid signature should be rejected.

; CHECK: intrinsic has incorrect number of args. Expected 1, but got 2
define void @test(float %a) {
  call float @llvm.ceil.f32(float %a, float %a)
  ret void
}
