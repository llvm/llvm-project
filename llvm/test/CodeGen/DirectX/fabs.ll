; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for abs are generated for float, half, and double.


; CHECK-LABEL: fabs_half
define noundef half @fabs_half(half noundef %a) {
entry:
  ; CHECK: call half @dx.op.unary.f16(i32 6, half %{{.*}})
  %elt.abs = call half @llvm.fabs.f16(half %a)
  ret half %elt.abs
}

; CHECK-LABEL: fabs_float
define noundef float @fabs_float(float noundef %a) {
entry:
; CHECK: call float @dx.op.unary.f32(i32 6, float %{{.*}})
  %elt.abs = call float @llvm.fabs.f32(float %a)
  ret float %elt.abs
}

; CHECK-LABEL: fabs_double
define noundef double @fabs_double(double noundef %a) {
entry:
; CHECK: call double @dx.op.unary.f64(i32 6, double %{{.*}})
  %elt.abs = call double @llvm.fabs.f64(double %a)
  ret double %elt.abs
}

declare half @llvm.fabs.f16(half)
declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
