; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; Make sure the intrinsic dx.saturate is to appropriate DXIL op for half/float/double data types.

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.6-unknown-shadermodel6.6-library"

; CHECK-LABEL: test_saturate_half
define noundef half @test_saturate_half(half noundef %p0) #0 {
entry:
  %p0.addr = alloca half, align 2
  store half %p0, ptr %p0.addr, align 2, !tbaa !4
  %0 = load half, ptr %p0.addr, align 2, !tbaa !4
  ; CHECK: %1 = call half @dx.op.unary.f16(i32 7, half %0)
  %hlsl.saturate = call half @llvm.dx.saturate.f16(half %0)
  ; CHECK: ret half %1
  ret half %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare half @llvm.dx.saturate.f16(half) #1

; CHECK-LABEL: test_saturate_float
define noundef float @test_saturate_float(float noundef %p0) #0 {
entry:
  %p0.addr = alloca float, align 4
  store float %p0, ptr %p0.addr, align 4, !tbaa !9
  %0 = load float, ptr %p0.addr, align 4, !tbaa !9
  ; CHECK: %1 = call float @dx.op.unary.f32(i32 7, float %0)
  %hlsl.saturate = call float @llvm.dx.saturate.f32(float %0)
  ; CHECK: ret float %1
  ret float %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare float @llvm.dx.saturate.f32(float) #1

; CHECK-LABEL: test_saturate_double
define noundef double @test_saturate_double(double noundef %p0) #0 {
entry:
  %p0.addr = alloca double, align 8
  store double %p0, ptr %p0.addr, align 8, !tbaa !11
  %0 = load double, ptr %p0.addr, align 8, !tbaa !11
  ; CHECK: %1 = call double @dx.op.unary.f64(i32 7, double %0)
  %hlsl.saturate = call double @llvm.dx.saturate.f64(double %0)
  ; CHECK: ret double %1
  ret double %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare double @llvm.dx.saturate.f64(double) #1

attributes #0 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0, !1}
!dx.valver = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 7}
!4 = !{!5, !5, i64 0}
!5 = !{!"half", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!6, !6, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"float", !6, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"double", !6, i64 0}
