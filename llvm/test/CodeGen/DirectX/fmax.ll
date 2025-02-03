; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for fmax are generated for half/float/double.

; CHECK-LABEL:test_fmax_half
define noundef half @test_fmax_half(half noundef %a, half noundef %b) {
entry:
; CHECK: call half @dx.op.binary.f16(i32 35, half %{{.*}}, half %{{.*}}) #[[#ATTR:]]
  %0 = call half @llvm.maxnum.f16(half %a, half %b)
  ret half %0
}

; CHECK-LABEL:test_fmax_float
define noundef float @test_fmax_float(float noundef %a, float noundef %b) {
entry:
; CHECK: call float @dx.op.binary.f32(i32 35, float %{{.*}}, float %{{.*}}) #[[#ATTR]]
  %0 = call float @llvm.maxnum.f32(float %a, float %b)
  ret float %0
}

; CHECK-LABEL:test_fmax_double
define noundef double @test_fmax_double(double noundef %a, double noundef %b) {
entry:
; CHECK: call double @dx.op.binary.f64(i32 35, double %{{.*}}, double %{{.*}}) #[[#ATTR]]
  %0 = call double @llvm.maxnum.f64(double %a, double %b)
  ret double %0
}

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}

declare half @llvm.maxnum.f16(half, half)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)
