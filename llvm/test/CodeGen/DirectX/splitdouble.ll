; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s
; RUN: opt -S --scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; Make sure DXILOpLowering is correctly generating the dxil op, with and without scalarizer.

; CHECK-LABEL: define noundef i32 @test_scalar_double_split
define noundef i32 @test_scalar_double_split(double noundef %D) local_unnamed_addr {
entry:
  ; CHECK: [[CALL:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double %D)
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  ; CHECK-NEXT:extractvalue %dx.types.splitdouble [[CALL]], {{[0-1]}}
  %hlsl.splitdouble = call { i32, i32 } @llvm.dx.splitdouble.i32(double %D)
  %0 = extractvalue { i32, i32 } %hlsl.splitdouble, 0
  %1 = extractvalue { i32, i32 } %hlsl.splitdouble, 1
  %add = add i32 %0, %1
  ret i32 %add
}
