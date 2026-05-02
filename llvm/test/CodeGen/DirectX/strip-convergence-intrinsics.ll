; RUN: llc %s -mtriple=dxil-pc-shadermodel6.3-library -o - | FileCheck %s

; Verify that convergence intrinsics and operand bundles are stripped
; during DXIL lowering pipeline.

; CHECK-LABEL: define float @test(
; CHECK-NOT: convergence
; CHECK: ret float

; CHECK-LABEL: define void @test_loop(
; CHECK-NOT: convergence
; CHECK: ret void

define float @test(float %a) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %1 = call float @llvm.dx.dot(float %a) [ "convergencectrl"(token %0) ]
  ret float %1
}

define void @test_loop(float %a) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  br label %loop

loop:
  %1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %0) ]
  call void @convergent_callee() [ "convergencectrl"(token %1) ]
  br i1 true, label %exit, label %loop

exit:
  ret void
}

declare void @convergent_callee() convergent
declare float @llvm.dx.dot(float)
declare token @llvm.experimental.convergence.entry()
declare token @llvm.experimental.convergence.loop()
