; RUN: opt -passes=instcombine -disable-output %s
; ModuleID = 'reduced.bc'
source_filename = "repro.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global i32 0, align 4, !guid !0
@b = dso_local local_unnamed_addr global i8 0, align 1, !guid !1
@a = dso_local local_unnamed_addr global i32 0, align 4, !guid !2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @d() local_unnamed_addr #0 !guid !12 {
entry:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @e(i32 noundef %g, i32 noundef %h, i32 noundef %i) local_unnamed_addr #0 !guid !13 {
entry:
  %cmp = icmp slt i32 %g, %h
  %cond = call i32 @llvm.smin.i32(i32 %g, i32 %i)
  %cond5 = select i1 %cmp, i32 0, i32 %cond
  ret i32 %cond5
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nounwind uwtable
define dso_local void @j() local_unnamed_addr #2 !guid !14 {
entry:
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %k.0 = phi i32 [ 893196426, %entry ], [ 826450915, %do.body ]
  %conv = sext i32 %k.0 to i64
  %shl = shl i64 %conv, 52
  %shr = ashr exact i64 %shl, 52
  %0 = trunc nsw i64 %shr to i32
  %conv1 = sub nsw i32 0, %0
  %add = add nsw i32 %0, 2
  %sub14 = sub nsw i32 2, %0
  %cmp.i = icmp slt i32 %conv1, %add
  %cond.i = call i32 @llvm.smin.i32(i32 %conv1, i32 %sub14)
  %cond5.i = select i1 %cmp.i, i32 0, i32 %cond.i
  store i32 %cond5.i, ptr @c, align 4, !tbaa !15
  %1 = load i8, ptr @b, align 1, !tbaa !16
  %conv15 = sext i8 %1 to i32
  %sub16 = sub nsw i32 826450915, %conv15
  %tobool.not = icmp eq i32 %sub16, 7
  br i1 %tobool.not, label %do.body, label %do.end, !llvm.loop !17

do.end:                                           ; preds = %do.body
  %2 = load i32, ptr @a, align 4, !tbaa !15
  ret void

; uselistorder directives
  uselistorder i32 %0, { 1, 0, 2 }
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #3

; uselistorder directives
uselistorder ptr @llvm.smin.i32, { 1, 0 }

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}
!llvm.errno.tbaa = !{!7}

!0 = !{i64 4014738744300571210}
!1 = !{i64 -1427730249719747694}
!2 = !{i64 -6289574019528802036}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 7, !"PIE Level", i32 2}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{!"clang version 24.0.0git (https://github.com/im-lunex/llvm-project 23e0cbe9a37fdd7addfd461ace6b480ad423e36b)"}
!7 = !{!8, !9, i64 0}
!8 = !{!"__libc_errno", !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{i64 -7709752385939146878}
!13 = !{i64 -642555945652099103}
!14 = !{i64 -2354099122118444234}
!15 = !{!9, !9, i64 0}
!16 = !{!10, !10, i64 0}
!17 = distinct !{!17, !18, !19}
!18 = !{!"llvm.loop.mustprogress"}
!19 = !{!"llvm.loop.unroll.disable"}
