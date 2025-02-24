; RUN: opt < %s -passes=memcpyopt,dse -S -verify-memoryssa | FileCheck %s
; The aim of this test is to check if MemCpyOpt pass merges alias tags
; after memcpy optimization

; ModuleID = 'FIRModule'
source_filename = "FIRModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@data_arr = internal unnamed_addr constant [31 x float] [float 0x3E68DA0CA0000000, float 0x3E692863A0000000, float 0x3E6AEF5000000000, float 0x3E6E2272C0000000, float 0x3E7271B720000000, float 0x3E777DA440000000, float 0x3E7E8C46C0000000, float 0x3E8458EFC0000000, float 0x3E8D0123C0000000, float 0x3E95E78260000000, float 0x3EA0AB7AC0000000, float 0x3EA89F4B40000000, float 0x3EB10FFB60000000, float 0x3EB5F1D140000000, float 0x3EBB435260000000, float 0x3EC0DE9700000000, float 0x3EC51B11A0000000, float 0x3ECA419FC0000000, float 0x3ED01B2B20000000, float 0x3ED3B9CEC0000000, float 0x3ED7028C40000000, float 0x3EDA60C320000000, float 0x3EDD54AD40000000, float 0x3EDF6E9F00000000, float 0x3EE130BB20000000, float 0x3EE4332400000000, float 0x3EE7575F80000000, float 0x3EE8088A60000000, float 0x3EE3B0AE60000000, float 0x3ED9BB6800000000, float 0x3ED9BB6800000000]

; CHECK-LABEL: @test(
; CHECK:         [[ARR_UNDER_TEST:%.*]] = alloca [31 x float], align 4
; CHECK:         store float 0x3E6AA51880000000, ptr [[ARR_UNDER_TEST]], align 4, !tbaa [[ARR_TAG:!.[0-9]+]]  
; CHECK-LABEL: init_loop:
; CHECK:           store float [[TMP0:%.*]], ptr [[TMP1:%.*]], align 4, !tbaa [[ARR_TAG]]
; CHECK-LABEL: loop:
; CHECK:           [[TMP2:%.*]] = getelementptr float, ptr [[ARR_UNDER_TEST]], i64 [[TMP3:%.*]]
; CHECK:           [[TMP4:%.*]] = load float, ptr [[TMP2]], align 4, !tbaa [[ARR_TAG]]
define void @test(ptr captures(none) %0, ptr readonly captures(none) %1, ptr readonly captures(none) %2, ptr readonly captures(none) %3) local_unnamed_addr #0 {
  %5 = alloca [32 x float], align 4
  %6 = alloca [31 x float], align 4
  %7 = alloca [31 x float], align 4
  %8 = load i32, ptr %2, align 4, !tbaa !4
  %9 = sext i32 %8 to i64
  %10 = load i32, ptr %3, align 4, !tbaa !10
  %11 = add i32 %10, 1
  %12 = sext i32 %11 to i64
  %13 = sub nsw i64 %12, %9
  %14 = tail call i64 @llvm.smax.i64(i64 %13, i64 -1)
  %15 = add nsw i64 %14, 1
  %16 = alloca float, i64 %15, align 4
  store float 0x3E6AA51880000000, ptr %7, align 4, !tbaa !12
  br label %init_loop

init_loop:
  %19 = phi float [ 0x3E68DA0CA0000000, %4 ], [ %22, %init_loop ]
  %indvars.iv = phi i64 [ 2, %4 ], [ %indvars.iv.next, %init_loop ]
  %20 = add nsw i64 %indvars.iv, -1
  %21 = getelementptr float, ptr @data_arr, i64 %20
  %22 = load float, ptr %21, align 4, !tbaa !15
  %23 = fsub contract float %22, %19
  %33 = getelementptr float, ptr %7, i64 %20
  store float %23, ptr %33, align 4, !tbaa !12
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 32
  br i1 %exitcond.not, label %.preheader55.preheader, label %init_loop

.preheader55.preheader:
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(124) %6, ptr noundef nonnull align 4 dereferenceable(124) %7, i64 124, i1 false), !tbaa !22
  %154 = icmp sgt i64 %13, -1
  br i1 %154, label %loop, label %._crit_edge56

loop:                                              ; preds = %.preheader, %211
  %indvars.iv73 = phi i64 [ 0, %.preheader55.preheader ], [ %indvars.iv.next74, %loop ]
  %indvars.iv.next74 = add nuw nsw i64 %indvars.iv73, 1
  %223 = getelementptr float, ptr %6, i64 %indvars.iv73
  %225 = load float, ptr %223, align 4, !tbaa !31
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
!4 = !{!5, !5, i64 0}
!5 = !{!"dummy arg data/param_1", !6, i64 0}
!6 = !{!"dummy arg data", !7, i64 0}
!7 = !{!"any data access", !8, i64 0}
!8 = !{!"any access", !9, i64 0}
!9 = !{!"Flang function root test"}
!10 = !{!11, !11, i64 0}
!11 = !{!"dummy arg data/param_2", !6, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"allocated data/test_array_a", !14, i64 0}
!14 = !{!"allocated data", !7, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"global data/data_arr", !17, i64 0}
!17 = !{!"global data", !7, i64 0}
!22 = !{!14, !14, i64 0}
!31 = !{!32, !32, i64 0}
!32 = !{!"allocated data/test_array_b", !14, i64 0}
