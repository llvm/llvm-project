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

; CHECK-LABEL: test_saturate_half2
define noundef <2 x half> @test_saturate_half2(<2 x half> noundef %p0) #0 {
entry:
  %p0.addr = alloca <2 x half>, align 4
  store <2 x half> %p0, ptr %p0.addr, align 4, !tbaa !8
  %0 = load <2 x half>, ptr %p0.addr, align 4, !tbaa !8
  ; CHECK: %1 = extractelement <2 x half> %0, i64 0
  ; CHECK-NEXT: %2 = call half @dx.op.unary.f16(i32 7, half %1)
  ; CHECK-NEXT: %3 = insertelement <2 x half> %0, half %2, i64 0
  ; CHECK-NEXT: %4 = extractelement <2 x half> %0, i64 1
  ; CHECK-NEXT: %5 = call half @dx.op.unary.f16(i32 7, half %4)
  ; CHECK-NEXT: %6 = insertelement <2 x half> %0, half %5, i64 1
  %hlsl.saturate = call <2 x half> @llvm.dx.saturate.v2f16(<2 x half> %0)
  ; CHECK: ret <2 x half> %6
  ret <2 x half> %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <2 x half> @llvm.dx.saturate.v2f16(<2 x half>) #1

; CHECK-LABEL: test_saturate_half3
define noundef <3 x half> @test_saturate_half3(<3 x half> noundef %p0) #0 {
entry:
  %p0.addr = alloca <3 x half>, align 8
  store <3 x half> %p0, ptr %p0.addr, align 8, !tbaa !8
  %0 = load <3 x half>, ptr %p0.addr, align 8, !tbaa !8
  ; CHECK: %1 = extractelement <3 x half> %0, i64 0
  ; CHECK-NEXT: %2 = call half @dx.op.unary.f16(i32 7, half %1)
  ; CHECK-NEXT: %3 = insertelement <3 x half> %0, half %2, i64 0
  ; CHECK-NEXT: %4 = extractelement <3 x half> %0, i64 1
  ; CHECK-NEXT: %5 = call half @dx.op.unary.f16(i32 7, half %4)
  ; CHECK-NEXT: %6 = insertelement <3 x half> %0, half %5, i64 1
  ; CHECK-NEXT: %7 = extractelement <3 x half> %0, i64 2
  ; CHECK-NEXT: %8 = call half @dx.op.unary.f16(i32 7, half %7)
  ; CHECK-NEXT: %9 = insertelement <3 x half> %0, half %8, i64 2
  %hlsl.saturate = call <3 x half> @llvm.dx.saturate.v3f16(<3 x half> %0)
  ; CHECK: ret <3 x half> %9
  ret <3 x half> %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <3 x half> @llvm.dx.saturate.v3f16(<3 x half>) #1

; CHECK-LABEL: test_saturate_half4
define noundef <4 x half> @test_saturate_half4(<4 x half> noundef %p0) #0 {
entry:
  %p0.addr = alloca <4 x half>, align 8
  store <4 x half> %p0, ptr %p0.addr, align 8, !tbaa !8
  %0 = load <4 x half>, ptr %p0.addr, align 8, !tbaa !8
  ; CHECK: %1 = extractelement <4 x half> %0, i64 0
  ; CHECK-NEXT: %2 = call half @dx.op.unary.f16(i32 7, half %1)
  ; CHECK-NEXT: %3 = insertelement <4 x half> %0, half %2, i64 0
  ; CHECK-NEXT: %4 = extractelement <4 x half> %0, i64 1
  ; CHECK-NEXT: %5 = call half @dx.op.unary.f16(i32 7, half %4)
  ; CHECK-NEXT: %6 = insertelement <4 x half> %0, half %5, i64 1
  ; CHECK-NEXT: %7 = extractelement <4 x half> %0, i64 2
  ; CHECK-NEXT: %8 = call half @dx.op.unary.f16(i32 7, half %7)
  ; CHECK-NEXT: %9 = insertelement <4 x half> %0, half %8, i64 2
  ; CHECK-NEXT: %10 = extractelement <4 x half> %0, i64 3
  ; CHECK-NEXT: %11 = call half @dx.op.unary.f16(i32 7, half %10)
  ; CHECK-NEXT: %12 = insertelement <4 x half> %0, half %11, i64 3
  %hlsl.saturate = call <4 x half> @llvm.dx.saturate.v4f16(<4 x half> %0)
  ; CHECK: ret <4 x half> %12
  ret <4 x half> %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x half> @llvm.dx.saturate.v4f16(<4 x half>) #1

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

; CHECK-LABEL: test_saturate_float2
define noundef <2 x float> @test_saturate_float2(<2 x float> noundef %p0) #0 {
entry:
  %p0.addr = alloca <2 x float>, align 8
  store <2 x float> %p0, ptr %p0.addr, align 8, !tbaa !8
  %0 = load <2 x float>, ptr %p0.addr, align 8, !tbaa !8
  ; CHECK: %1 = extractelement <2 x float> %0, i64 0
  ; CHECK-NEXT: %2 = call float @dx.op.unary.f32(i32 7, float %1)
  ; CHECK-NEXT: %3 = insertelement <2 x float> %0, float %2, i64 0
  ; CHECK-NEXT: %4 = extractelement <2 x float> %0, i64 1
  ; CHECK-NEXT: %5 = call float @dx.op.unary.f32(i32 7, float %4)
  ; CHECK-NEXT: %6 = insertelement <2 x float> %0, float %5, i64 1
  %hlsl.saturate = call <2 x float> @llvm.dx.saturate.v2f32(<2 x float> %0)
  ; CHECK: ret <2 x float> %6
  ret <2 x float> %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <2 x float> @llvm.dx.saturate.v2f32(<2 x float>) #1

; CHECK-LABEL: test_saturate_float3
define noundef <3 x float> @test_saturate_float3(<3 x float> noundef %p0) #0 {
entry:
  %p0.addr = alloca <3 x float>, align 16
  store <3 x float> %p0, ptr %p0.addr, align 16, !tbaa !8
  %0 = load <3 x float>, ptr %p0.addr, align 16, !tbaa !8
  ; CHECK: %1 = extractelement <3 x float> %0, i64 0
  ; CHECK-NEXT: %2 = call float @dx.op.unary.f32(i32 7, float %1)
  ; CHECK-NEXT: %3 = insertelement <3 x float> %0, float %2, i64 0
  ; CHECK-NEXT: %4 = extractelement <3 x float> %0, i64 1
  ; CHECK-NEXT: %5 = call float @dx.op.unary.f32(i32 7, float %4)
  ; CHECK-NEXT: %6 = insertelement <3 x float> %0, float %5, i64 1
  ; CHECK-NEXT: %7 = extractelement <3 x float> %0, i64 2
  ; CHECK-NEXT: %8 = call float @dx.op.unary.f32(i32 7, float %7)
  ; CHECK-NEXT: %9 = insertelement <3 x float> %0, float %8, i64 2
  %hlsl.saturate = call <3 x float> @llvm.dx.saturate.v3f32(<3 x float> %0)
  ; CHECK: ret <3 x float> %9
  ret <3 x float> %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <3 x float> @llvm.dx.saturate.v3f32(<3 x float>) #1

; CHECK-LABEL: test_saturate_float4
define noundef <4 x float> @test_saturate_float4(<4 x float> noundef %p0) #0 {
entry:
  %p0.addr = alloca <4 x float>, align 16
  store <4 x float> %p0, ptr %p0.addr, align 16, !tbaa !8
  %0 = load <4 x float>, ptr %p0.addr, align 16, !tbaa !8
  ; CHECK: %1 = extractelement <4 x float> %0, i64 0
  ; CHECK-NEXT: %2 = call float @dx.op.unary.f32(i32 7, float %1)
  ; CHECK-NEXT: %3 = insertelement <4 x float> %0, float %2, i64 0
  ; CHECK-NEXT: %4 = extractelement <4 x float> %0, i64 1
  ; CHECK-NEXT: %5 = call float @dx.op.unary.f32(i32 7, float %4)
  ; CHECK-NEXT: %6 = insertelement <4 x float> %0, float %5, i64 1
  ; CHECK-NEXT: %7 = extractelement <4 x float> %0, i64 2
  ; CHECK-NEXT: %8 = call float @dx.op.unary.f32(i32 7, float %7)
  ; CHECK-NEXT: %9 = insertelement <4 x float> %0, float %8, i64 2
  ; CHECK-NEXT: %10 = extractelement <4 x float> %0, i64 3
  ; CHECK-NEXT: %11 = call float @dx.op.unary.f32(i32 7, float %10)
  ; CHECK-NEXT: %12 = insertelement <4 x float> %0, float %11, i64 3
  %hlsl.saturate = call <4 x float> @llvm.dx.saturate.v4f32(<4 x float> %0)
  ; CHECK: ret <4 x float> %12
  ret <4 x float> %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x float> @llvm.dx.saturate.v4f32(<4 x float>) #1

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

; CHECK-LABEL: test_saturate_double2
define noundef <2 x double> @test_saturate_double2(<2 x double> noundef %p0) #0 {
entry:
  %p0.addr = alloca <2 x double>, align 16
  store <2 x double> %p0, ptr %p0.addr, align 16, !tbaa !8
  %0 = load <2 x double>, ptr %p0.addr, align 16, !tbaa !8
  ; CHECK: %1 = extractelement <2 x double> %0, i64 0
  ; CHECK-NEXT: %2 = call double @dx.op.unary.f64(i32 7, double %1)
  ; CHECK-NEXT: %3 = insertelement <2 x double> %0, double %2, i64 0
  ; CHECK-NEXT: %4 = extractelement <2 x double> %0, i64 1
  ; CHECK-NEXT: %5 = call double @dx.op.unary.f64(i32 7, double %4)
  ; CHECK-NEXT: %6 = insertelement <2 x double> %0, double %5, i64 1
  %hlsl.saturate = call <2 x double> @llvm.dx.saturate.v2f64(<2 x double> %0)
  ; CHECK: ret <2 x double> %6
  ret <2 x double> %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <2 x double> @llvm.dx.saturate.v2f64(<2 x double>) #1

; CHECK-LABEL: test_saturate_double3
define noundef <3 x double> @test_saturate_double3(<3 x double> noundef %p0) #0 {
entry:
  %p0.addr = alloca <3 x double>, align 32
  store <3 x double> %p0, ptr %p0.addr, align 32, !tbaa !8
  %0 = load <3 x double>, ptr %p0.addr, align 32, !tbaa !8
  ; CHECK: %1 = extractelement <3 x double> %0, i64 0
  ; CHECK-NEXT: %2 = call double @dx.op.unary.f64(i32 7, double %1)
  ; CHECK-NEXT: %3 = insertelement <3 x double> %0, double %2, i64 0
  ; CHECK-NEXT: %4 = extractelement <3 x double> %0, i64 1
  ; CHECK-NEXT: %5 = call double @dx.op.unary.f64(i32 7, double %4)
  ; CHECK-NEXT: %6 = insertelement <3 x double> %0, double %5, i64 1
  ; CHECK-NEXT: %7 = extractelement <3 x double> %0, i64 2
  ; CHECK-NEXT: %8 = call double @dx.op.unary.f64(i32 7, double %7)
  ; CHECK-NEXT: %9 = insertelement <3 x double> %0, double %8, i64 2
  %hlsl.saturate = call <3 x double> @llvm.dx.saturate.v3f64(<3 x double> %0)
  ; CHECK: ret <3 x double> %9
  ret <3 x double> %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <3 x double> @llvm.dx.saturate.v3f64(<3 x double>) #1

; CHECK-LABEL: test_saturate_double4
define noundef <4 x double> @test_saturate_double4(<4 x double> noundef %p0) #0 {
entry:
  %p0.addr = alloca <4 x double>, align 32
  store <4 x double> %p0, ptr %p0.addr, align 32, !tbaa !8
  %0 = load <4 x double>, ptr %p0.addr, align 32, !tbaa !8
  ; CHECK: %1 = extractelement <4 x double> %0, i64 0
  ; CHECK-NEXT: %2 = call double @dx.op.unary.f64(i32 7, double %1)
  ; CHECK-NEXT: %3 = insertelement <4 x double> %0, double %2, i64 0
  ; CHECK-NEXT: %4 = extractelement <4 x double> %0, i64 1
  ; CHECK-NEXT: %5 = call double @dx.op.unary.f64(i32 7, double %4)
  ; CHECK-NEXT: %6 = insertelement <4 x double> %0, double %5, i64 1
  ; CHECK-NEXT: %7 = extractelement <4 x double> %0, i64 2
  ; CHECK-NEXT: %8 = call double @dx.op.unary.f64(i32 7, double %7)
  ; CHECK-NEXT: %9 = insertelement <4 x double> %0, double %8, i64 2
  ; CHECK-NEXT: %10 = extractelement <4 x double> %0, i64 3
  ; CHECK-NEXT: %11 = call double @dx.op.unary.f64(i32 7, double %10)
  ; CHECK-NEXT: %12 = insertelement <4 x double> %0, double %11, i64 3
  %hlsl.saturate = call <4 x double> @llvm.dx.saturate.v4f64(<4 x double> %0)
  ; CHECK: ret <4 x double> %12
  ret <4 x double> %hlsl.saturate
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x double> @llvm.dx.saturate.v4f64(<4 x double>) #1

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
