; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s
; Make sure the intrinsic dx.saturate is to appropriate DXIL op for half/float/double data types.

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.6-unknown-shadermodel6.6-library"

; DXIL operation saturate does not support i32 overload
; CHECK: invalid intrinsic signature

define noundef i32 @test_saturate_i32(i32 noundef %p0) #0 {
entry:
  %hlsl.saturate = call i32 @llvm.dx.saturate.i32(i32 %p0)
  ret i32 %hlsl.saturate
}
