; RUN: opt -passes='function(scalarizer),module(dxil-op-lower,dxil-intrinsic-expansion)' -S -mtriple=dxil-pc-shadermodel6.3-pixel %s | FileCheck %s

; CHECK-LABEL: define void @test_scalar
; CHECK: call void @dx.op.discard(i32 82, i1 %0)
;
define void @test_scalar(float noundef %p) #0 {
entry:
  %0 = fcmp olt float %p, 0.000000e+00
  call void @llvm.dx.discard(i1 %0)
  ret void
}

; CHECK-LABEL: define void @test_vector
; CHECK:       [[EXTR0:%.*]] = extractelement <4 x i1> [[INP:%.*]], i64 0
; CHECK-NEXT:  [[EXTR1:%.*]] = extractelement <4 x i1> [[INP:%.*]], i64 1
; CHECK-NEXT:  [[OR1:%.*]] = or i1 [[EXTR0]], [[EXTR1]]
; CHECK-NEXT:  [[EXTR2:%.*]] = extractelement <4 x i1> [[INP:%.*]], i64 2
; CHECK-NEXT:  [[OR2:%.*]] = or i1 [[OR1]], [[EXTR2]]
; CHECK-NEXT:  [[EXTR3:%.*]] = extractelement <4 x i1> [[INP:%.*]], i64 3
; CHECK-NEXT:  [[OR3:%.*]] = or i1 [[OR2]], [[EXTR3]]
; CHECK-NEXT:  call void @dx.op.discard(i32 82, i1 [[OR3]])
;
define void @test_vector(<4 x float> noundef %p) #0 {
entry:
  %0 = fcmp olt <4 x float> %p, zeroinitializer
  %1 = call i1 @llvm.dx.any.v4i1(<4 x i1> %0)
  call void @llvm.dx.discard(i1 %1)
  ret void
}

declare void @llvm.dx.discard(i1)
