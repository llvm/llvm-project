; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation saturate does not support i32 overload
; CHECK: invalid intrinsic signature

define noundef i32 @test_saturate_i32(i32 noundef %p0) #0 {
entry:
  %hlsl.saturate = call i32 @llvm.dx.saturate.i32(i32 %p0)
  ret i32 %hlsl.saturate
}
