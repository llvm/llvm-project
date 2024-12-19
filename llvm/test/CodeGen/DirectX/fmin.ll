; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for fmin are generated for half/float/double.

; CHECK-LABEL:test_fmin_half
define noundef half @test_fmin_half(half noundef %a, half noundef %b) {
entry:
; CHECK: call half @dx.op.binary.f16(i32 36, half %{{.*}}, half %{{.*}})
  %0 = call half @llvm.minnum.f16(half %a, half %b)
  ret half %0
}

; CHECK-LABEL:test_fmin_float
define noundef float @test_fmin_float(float noundef %a, float noundef %b) {
entry:
; CHECK: call float @dx.op.binary.f32(i32 36, float %{{.*}}, float %{{.*}})
  %0 = call float @llvm.minnum.f32(float %a, float %b)
  ret float %0
}

; CHECK-LABEL:test_fmin_double
define noundef double @test_fmin_double(double noundef %a, double noundef %b) {
entry:
; CHECK: call double @dx.op.binary.f64(i32 36, double %{{.*}}, double %{{.*}})
  %0 = call double @llvm.minnum.f64(double %a, double %b)
  ret double %0
}

declare half @llvm.minnum.f16(half, half)
declare float @llvm.minnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)
