; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for fabs are generated for half, float and double.

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.5-compute"

@"?h1@@3$halff@A" = local_unnamed_addr global float 0.000000e+00, align 4
@"?h2@@3$halff@A" = local_unnamed_addr global float 0.000000e+00, align 4
@"?f1@@3MA" = local_unnamed_addr global float 0.000000e+00, align 4
@"?f2@@3MA" = local_unnamed_addr global float 0.000000e+00, align 4
@"?d1@@3NA" = local_unnamed_addr global double 0.000000e+00, align 8
@"?d2@@3NA" = local_unnamed_addr global double 0.000000e+00, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define void @"?test_half@@YAXXZ"() local_unnamed_addr #1 {
entry:
  %0 = load float, ptr @"?h2@@3$halff@A", align 4, !tbaa !4
  ; CHECK: %1 = tail call float @dx.op.unary.f32(i32 6, float %0)
  %elt.abs = tail call float @llvm.fabs.f32(float %0)
  ; CHECK: store float %1, ptr @"?h1@@3$halff@A", align 4, !tbaa !4
  store float %elt.abs, ptr @"?h1@@3$halff@A", align 4, !tbaa !4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define void @"?test_float@@YAXXZ"() local_unnamed_addr #1 {
entry:
  %0 = load float, ptr @"?f2@@3MA", align 4, !tbaa !8
  ; CHECK: %1 = tail call float @dx.op.unary.f32(i32 6, float %0)
  %elt.abs = tail call float @llvm.fabs.f32(float %0)
  ; CHECK: store float %1, ptr @"?f1@@3MA", align 4, !tbaa !8
  store float %elt.abs, ptr @"?f1@@3MA", align 4, !tbaa !8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define void @"?test_double@@YAXXZ"() local_unnamed_addr #1 {
entry:
  %0 = load double, ptr @"?d2@@3NA", align 8, !tbaa !10
  ; CHECK: %1 = tail call double @dx.op.unary.f64(i32 6, double %0)
  %elt.abs = tail call double @llvm.fabs.f64(double %0)
  ; CHECK: store double %1, ptr @"?d1@@3NA", align 8, !tbaa !10
  store double %elt.abs, ptr @"?d1@@3NA", align 8, !tbaa !10
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "frame-pointer"="all" "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!dx.valver = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 7}
!3 = !{!"clang version 18.0.0git (git@github.com:somefork/llvm-project.git someSHA)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"half", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !6, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !6, i64 0}