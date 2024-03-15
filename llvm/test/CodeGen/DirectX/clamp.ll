; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for clamp/uclamp are generated for half/float/double/i16/i32/i64.

; CHECK-LABEL:test_clamp_i16
define noundef i16 @test_clamp_i16(i16 noundef %a, i16 noundef %b, i16 noundef %c) {
entry:
; CHECK: call i16 @dx.op.binary.i16(i32 37, i16 %{{.*}}, i16 %{{.*}})
; CHECK: call i16 @dx.op.binary.i16(i32 38, i16 %{{.*}}, i16 %{{.*}})
  %0 = call i16 @llvm.dx.clamp.i16(i16 %a, i16 %b, i16 %c)
  ret i16 %0
}

; CHECK-LABEL:test_clamp_i32
define noundef i32 @test_clamp_i32(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
; CHECK: call i32 @dx.op.binary.i32(i32 37, i32 %{{.*}}, i32 %{{.*}})
; CHECK: call i32 @dx.op.binary.i32(i32 38, i32 %{{.*}}, i32 %{{.*}})
  %0 = call i32 @llvm.dx.clamp.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %0
}

; CHECK-LABEL:test_clamp_i64
define noundef i64 @test_clamp_i64(i64 noundef %a, i64 noundef %b, i64 noundef %c) {
entry:
; CHECK: call i64 @dx.op.binary.i64(i32 37, i64 %a, i64 %b)
; CHECK: call i64 @dx.op.binary.i64(i32 38, i64 %{{.*}}, i64 %c)
  %0 = call i64 @llvm.dx.clamp.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %0
}

; CHECK-LABEL:test_clamp_half
define noundef half @test_clamp_half(half noundef %a, half noundef %b, half noundef %c) {
entry:
; CHECK: call half @dx.op.binary.f16(i32 35, half %{{.*}}, half %{{.*}})
; CHECK: call half @dx.op.binary.f16(i32 36, half %{{.*}}, half %{{.*}})
  %0 = call half @llvm.dx.clamp.f16(half %a, half %b, half %c)
  ret half %0
}

; CHECK-LABEL:test_clamp_float
define noundef float @test_clamp_float(float noundef %a, float noundef %b, float noundef %c) {
entry:
; CHECK: call float @dx.op.binary.f32(i32 35, float %{{.*}}, float %{{.*}})
; CHECK: call float @dx.op.binary.f32(i32 36, float %{{.*}}, float %{{.*}})
  %0 = call float @llvm.dx.clamp.f32(float %a, float %b, float %c)
  ret float %0
}

; CHECK-LABEL:test_clamp_double
define noundef double @test_clamp_double(double noundef %a, double noundef %b, double noundef %c) {
entry:
; CHECK: call double @dx.op.binary.f64(i32 35, double %{{.*}}, double %{{.*}})
; CHECK: call double @dx.op.binary.f64(i32 36, double %{{.*}}, double %{{.*}})
  %0 = call double @llvm.dx.clamp.f64(double %a, double %b, double %c)
  ret double %0
}

; CHECK-LABEL:test_uclamp_i16
define noundef i16 @test_uclamp_i16(i16 noundef %a, i16 noundef %b, i16 noundef %c) {
entry:
; CHECK: call i16 @dx.op.binary.i16(i32 39, i16 %{{.*}}, i16 %{{.*}})
; CHECK: call i16 @dx.op.binary.i16(i32 40, i16 %{{.*}}, i16 %{{.*}})
  %0 = call i16 @llvm.dx.uclamp.i16(i16 %a, i16 %b, i16 %c)
  ret i16 %0
}

; CHECK-LABEL:test_uclamp_i32
define noundef i32 @test_uclamp_i32(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
; CHECK: call i32 @dx.op.binary.i32(i32 39, i32 %{{.*}}, i32 %{{.*}})
; CHECK: call i32 @dx.op.binary.i32(i32 40, i32 %{{.*}}, i32 %{{.*}})
  %0 = call i32 @llvm.dx.uclamp.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %0
}

; CHECK-LABEL:test_uclamp_i64
define noundef i64 @test_uclamp_i64(i64 noundef %a, i64 noundef %b, i64 noundef %c) {
entry:
; CHECK: call i64 @dx.op.binary.i64(i32 39, i64 %a, i64 %b)
; CHECK: call i64 @dx.op.binary.i64(i32 40, i64 %{{.*}}, i64 %c)
  %0 = call i64 @llvm.dx.uclamp.i64(i64 %a, i64 %b, i64 %c)
  ret i64 %0
}

declare half @llvm.dx.clamp.f16(half, half, half)
declare float @llvm.dx.clamp.f32(float, float, float)
declare double @llvm.dx.clamp.f64(double, double, double)
declare i16 @llvm.dx.clamp.i16(i16, i16, i16)
declare i32 @llvm.dx.clamp.i32(i32, i32, i32)
declare i64 @llvm.dx.clamp.i64(i64, i64, i64)
declare i16 @llvm.dx.uclamp.i16(i16, i16, i16)
declare i32 @llvm.dx.uclamp.i32(i32, i32, i32)
declare i64 @llvm.dx.uclamp.i64(i64, i64, i64)
