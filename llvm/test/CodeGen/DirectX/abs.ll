; Make sure dxil operation function calls for abs are appropriately strength reduced for int and int64_t.
; RUN: opt -S -dxil-strength-reduce < %s | FileCheck %s -check-prefix=TEST_SR

; Make sure output of strength reduction pass is lowered to DXIL code as expected.
; RUN: opt -S -dxil-strength-reduce -dxil-op-lower < %s | FileCheck %s -check-prefix=TEST_SR_OP_LOWER


target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.5-compute"

@"?a@@3HA" = local_unnamed_addr global i32 0, align 4
@"?b@@3HA" = local_unnamed_addr global i32 0, align 4
@"?c@@3JA" = local_unnamed_addr global i64 0, align 8
@"?d@@3JA" = local_unnamed_addr global i64 0, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define void @"?test_i32@@YAXXZ"() local_unnamed_addr #1 {
entry:
  %0 = load i32, ptr @"?b@@3HA", align 4, !tbaa !4
  ; TEST_SR:%NegArg = sub i32 0, %0
  ; TEST_SR-NEXT: %IMax = tail call i32 @llvm.smax.i32(i32 %0, i32 %NegArg)
  ; TEST_SR_OP_LOWER: %NegArg = sub i32 0, %0
  ; TEST_SR_OP_LOWER-NEXT:%1 = tail call i32 @dx.op.binary.i32(i32 37, i32 %0, i32 %NegArg)
  %elt.abs = tail call i32 @llvm.abs.i32(i32 %0, i1 false)
  ; TEST_SR: store i32 %IMax, ptr @"?a@@3HA", align 4, !tbaa !4
  ; TEST_SR_OP_LOWER: store i32 %1, ptr @"?a@@3HA", align 4, !tbaa !4
  store i32 %elt.abs, ptr @"?a@@3HA", align 4, !tbaa !4
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.abs.i32(i32, i1 immarg) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none)
define void @"?test_i64@@YAXI@Z"(i32 noundef %GI) local_unnamed_addr #1 {
entry:
  %0 = load i64, ptr @"?d@@3JA", align 8, !tbaa !8
  ; TEST_SR: %NegArg = sub i64 0, %0
  ; TEST_SR-NEXT: %IMax = tail call i64 @llvm.smax.i64(i64 %0, i64 %NegArg)
  ; TEST_SR_OP_LOWER: %NegArg = sub i64 0, %0
  ; TEST_SR_OP_LOWER-NEXT: %1 = tail call i64 @dx.op.binary.i64(i32 37, i64 %0, i64 %NegArg)
  %elt.abs = tail call i64 @llvm.abs.i64(i64 %0, i1 false)
  ; TEST_SR: store i64 %IMax, ptr @"?c@@3JA", align 8, !tbaa !8
  ; TEST_SR_OP_LOWER: store i64 %1, ptr @"?c@@3JA", align 8, !tbaa !8
  store i64 %elt.abs, ptr @"?c@@3JA", align 8, !tbaa !8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.abs.i64(i64, i1 immarg) #2

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
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"long", !6, i64 0}