; RUN: opt < %s -passes=memcpyopt,dse -S -verify-memoryssa | FileCheck %s
; The aim of this test is to check if MemCpyOpt pass merges alias tags
; after memcpy optimization

; High level overview of this test
; Input:
; function test() {
;   //declaration of local arrays a and b
;   //initialization of array b in init_loop
;   //initialization of array a -> copy of array b
;   //use array a in loop
; }
;
; Expected output after optimization:
; function test() {
;   //declaration of local array b
;   //initialization of array b in init_loop
;   //use array b in loop
; }

; ModuleID = 'FIRModule'
source_filename = "FIRModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @test(
; CHECK:         [[ARR_UNDER_TEST:%.*]] = alloca [31 x float], align 4
; CHECK-LABEL: init_loop:
; CHECK:         [[TMP0:%.*]] = getelementptr float, ptr [[ARR_UNDER_TEST]],
; CHECK:         store float 0x3E6AA51880000000, ptr [[TMP0]], align 4, !tbaa [[ARR_TAG:![0-9]+]]
; CHECK-LABEL: loop:
; CHECK:         [[TMP2:%.*]] = getelementptr float, ptr [[ARR_UNDER_TEST]], i64 [[TMP3:%.*]]
; CHECK:         [[TMP4:%.*]] = load float, ptr [[TMP2]], align 4, !tbaa [[ARR_TAG]]

define void @test() local_unnamed_addr #0 {
  %test_array_a = alloca [31 x float], align 4
  %test_array_b = alloca [31 x float], align 4
  br label %init_loop

init_loop:
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %init_loop ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %1 = getelementptr float, ptr %test_array_b, i64 %indvars.iv
  store float 0x3E6AA51880000000, ptr %1, align 4, !tbaa !12
  %exitcond.not = icmp eq i64 %indvars.iv.next, 32
  br i1 %exitcond.not, label %.preheader55.preheader, label %init_loop

.preheader55.preheader:
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(124) %test_array_a, ptr noundef nonnull align 4 dereferenceable(124) %test_array_b, i64 124, i1 false)
  br label %loop

loop:                                              ; preds = %.preheader, %211
  %indvars.iv73 = phi i64 [ 0, %.preheader55.preheader ], [ %indvars.iv.next74, %loop ]
  %indvars.iv.next74 = add nuw nsw i64 %indvars.iv73, 1
  %2 = getelementptr float, ptr %test_array_a, i64 %indvars.iv73
  %3 = load float, ptr %2, align 4, !tbaa !31
  %exitcond76.not = icmp eq i64 %indvars.iv.next74, 32
  br i1 %exitcond76.not, label %loop, label %._crit_edge56

._crit_edge56:                                    ; preds = %loop, %._crit_edge
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

attributes #0 = { "target-cpu"="x86-64" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"flang version 21.0.0 (https://github.com/llvm/llvm-project.git 4d79f420ce5b5100f72f720eab2d3881f97abd0d)"}
!7 = !{!"any data access", !8, i64 0}
!8 = !{!"any access", !9, i64 0}
!9 = !{!"Flang function root test"}
!12 = !{!13, !13, i64 0}
!13 = !{!"allocated data/test_array_a", !14, i64 0}
!14 = !{!"allocated data", !7, i64 0}
!31 = !{!32, !32, i64 0}
!32 = !{!"allocated data/test_array_b", !14, i64 0}
