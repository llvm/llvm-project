; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for fabs are generated
; for half, float and double.

define noundef half @test_fabs_half(half noundef %a) #0 {
entry:
  ; CHECK: call half @dx.op.unary.f16(i32 6, half %{{.*}})
  %1 = call half @llvm.fabs.f16(half %a)
  ret half %1
}

define noundef float @test_fabs_float(float noundef %a) #0 {
entry:
  ; CHECK: call float @dx.op.unary.f32(i32 6, float %{{.*}})
  %1 = call float @llvm.fabs.f32(float %a)
  ret float %1
}

define noundef double @test_fabs_double(double noundef %a) #0 {
entry:
  ; CHECK: call double @dx.op.unary.f64(i32 6, double %{{.*}})
  %1 = call double @llvm.fabs.f64(double %a)
  ret double %1
}
