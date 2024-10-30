; RUN: opt -S -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-pixel %s | FileCheck %s

; CHECK-LABEL: define void @test_dxil_lowering
; CHECK: call void @dx.op.discard(i32 82, i1 %0)
;
define void @test_dxil_lowering(float noundef %p) #0 {
entry:
  %0 = fcmp olt float %p, 0.000000e+00
  call void @llvm.dx.clip(i1 %0)
  ret void
}
