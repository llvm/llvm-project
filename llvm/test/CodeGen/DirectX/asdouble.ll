; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Test that for scalar and vector inputs, asdouble maps down to the makeDouble
; DirectX op

define noundef double @asdouble_scalar(i32 noundef %low, i32 noundef %high) {
; CHECK: call double @dx.op.makeDouble.f64(i32 101, i32 %low, i32 %high)
  %ret = call double @llvm.dx.asdouble.i32(i32 %low, i32 %high)
  ret double %ret
}

declare double @llvm.dx.asdouble.i32(i32, i32)

define noundef <3 x double> @asdouble_vec(<3 x i32> noundef %low, <3 x i32> noundef %high) {
; CHECK: call double @dx.op.makeDouble.f64(i32 101, i32 %low.i0, i32 %high.i0)
; CHECK: call double @dx.op.makeDouble.f64(i32 101, i32 %low.i1, i32 %high.i1)
; CHECK: call double @dx.op.makeDouble.f64(i32 101, i32 %low.i2, i32 %high.i2)
  %ret = call <3 x double> @llvm.dx.asdouble.v3i32(<3 x i32> %low, <3 x i32> %high)
  ret <3 x double> %ret
}

declare <3 x double> @llvm.dx.asdouble.v3i32(<3 x i32>, <3 x i32>)
