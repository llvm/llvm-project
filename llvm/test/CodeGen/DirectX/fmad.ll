; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for round are generated for float and half.
; CHECK:call half @dx.op.tertiary.f16(i32 46, half %{{.*}}, half %{{.*}}, half %{{.*}})
; CHECK:call float @dx.op.tertiary.f32(i32 46, float %{{.*}}, float %{{.*}}, float %{{.*}})
; CHECK:call double @dx.op.tertiary.f64(i32 46, double %{{.*}}, double %{{.*}}, double %{{.*}})


target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"

; Function Attrs: noinline nounwind optnone
define noundef half @fmad_half(half noundef %p0, half noundef %p1, half noundef %p2) #0 {
entry:
  %p2.addr = alloca half, align 2
  %p1.addr = alloca half, align 2
  %p0.addr = alloca half, align 2
  store half %p2, ptr %p2.addr, align 2
  store half %p1, ptr %p1.addr, align 2
  store half %p0, ptr %p0.addr, align 2
  %0 = load half, ptr %p0.addr, align 2
  %1 = load half, ptr %p1.addr, align 2
  %2 = load half, ptr %p2.addr, align 2
  %dx.fmad = call half @llvm.fmuladd.f16(half %0, half %1, half %2)
  ret half %dx.fmad
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare half @llvm.fmuladd.f16(half, half, half) #2

; Function Attrs: noinline nounwind optnone
define noundef float @fmad_float(float noundef %p0, float noundef %p1, float noundef %p2) #0 {
entry:
  %p2.addr = alloca float, align 4
  %p1.addr = alloca float, align 4
  %p0.addr = alloca float, align 4
  store float %p2, ptr %p2.addr, align 4
  store float %p1, ptr %p1.addr, align 4
  store float %p0, ptr %p0.addr, align 4
  %0 = load float, ptr %p0.addr, align 4
  %1 = load float, ptr %p1.addr, align 4
  %2 = load float, ptr %p2.addr, align 4
  %dx.fmad = call float @llvm.fmuladd.f32(float %0, float %1, float %2)
  ret float %dx.fmad
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #2

; Function Attrs: noinline nounwind optnone
define noundef double @fmad_double(double noundef %p0, double noundef %p1, double noundef %p2) #0 {
entry:
  %p2.addr = alloca double, align 8
  %p1.addr = alloca double, align 8
  %p0.addr = alloca double, align 8
  store double %p2, ptr %p2.addr, align 8
  store double %p1, ptr %p1.addr, align 8
  store double %p0, ptr %p0.addr, align 8
  %0 = load double, ptr %p0.addr, align 8
  %1 = load double, ptr %p1.addr, align 8
  %2 = load double, ptr %p2.addr, align 8
  %dx.fmad = call double @llvm.fmuladd.f64(double %0, double %1, double %2)
  ret double %dx.fmad
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2
