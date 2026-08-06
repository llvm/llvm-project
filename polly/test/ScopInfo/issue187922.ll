; RUN: opt %loadNPMPolly '-passes=polly-custom<scops>' -polly-print-scops -disable-output < %s 2>&1 | FileCheck %s
;
; https://github.com/llvm/llvm-project/issues/187922
;
; The assumption 'p_2_loaded_from_var_0 <= 91' only holds if not 'p_2_loaded_from_var_0 >= 128'. Translation: if (trunc i64 %1 to i8) does not truncate any 1 bits, then we know that %1 <= 91.
; 'p_2_loaded_from_var_0 >= 128' must be in Invalid Context so we generate a RTC for it.
; 'p_2_loaded_from_var_0 <= 91' can only be in DefinedBehaviorContext, but not in "Context:" because "Context:" is used to gist AssumedContext and InvalidContext which would remove the required RTC: because %1 <= 91, the check if %1 >= 128 must not be optimized away.

; CHECK:      Context:
; CHECK-NEXT:   [p_0_loaded_from_var_3, p_1, p_2_loaded_from_var_0, p_3_loaded_from_var_7] ->  {  : -2147483648 <= p_0_loaded_from_var_3 <= 2147483647 and -1 <= p_1 <= 0 and -9223372036854775808 <= p_2_loaded_from_var_0 <= 9223372036854775807 and 0 <= p_3_loaded_from_var_7 <= 1 }
; CHECK:      Assumed Context:
; CHECK-NEXT:   [p_0_loaded_from_var_3, p_1, p_2_loaded_from_var_0, p_3_loaded_from_var_7] -> {  :  }
; CHECK:      Invalid Context:
; CHECK-NEXT:   [p_0_loaded_from_var_3, p_1, p_2_loaded_from_var_0, p_3_loaded_from_var_7] -> {  : p_2_loaded_from_var_0 <= -129 or p_2_loaded_from_var_0 >= 128 or (p_0_loaded_from_var_3 < 0 and p_2_loaded_from_var_0 >= 5490102402889819 - p_1) or (p_0_loaded_from_var_3 < 0 and p_2_loaded_from_var_0 <= 69) or (p_0_loaded_from_var_3 > 0 and p_2_loaded_from_var_0 >= 5490102402889819 - p_1) or (p_0_loaded_from_var_3 > 0 and p_2_loaded_from_var_0 <= 69) or (p_0_loaded_from_var_3 = 0 and p_3_loaded_from_var_7 = 1 and p_2_loaded_from_var_0 >= 5490102402889819 - p_1) or (p_0_loaded_from_var_3 = 0 and p_3_loaded_from_var_7 = 1 and p_2_loaded_from_var_0 <= 69) }
; CHECK:      Defined Behavior Context:
; CHECK-NEXT:   [p_0_loaded_from_var_3, p_1, p_2_loaded_from_var_0, p_3_loaded_from_var_7] -> {  : -1 <= p_1 <= 0 and -128 <= p_2_loaded_from_var_0 <= 91 and ((0 < p_0_loaded_from_var_3 <= 2147483647 and p_2_loaded_from_var_0 >= 70 and 0 <= p_3_loaded_from_var_7 <= 1) or (p_0_loaded_from_var_3 >= -2147483648 and p_2_loaded_from_var_0 >= 70 and p_3_loaded_from_var_7 > p_0_loaded_from_var_3 and 0 <= p_3_loaded_from_var_7 <= 1) or (p_0_loaded_from_var_3 = 0 and p_3_loaded_from_var_7 = 0)) }
; CHECK:      p0: %4
; CHECK:      p1: ((true + %tobool32.not) umin_seq (true + %.not))
; CHECK:      p2: %1
; CHECK:      p3: %3

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@var_0 = external dso_local local_unnamed_addr global i64, align 8
@var_3 = external dso_local local_unnamed_addr global i32, align 4
@var_4 = external dso_local local_unnamed_addr global i8, align 1
@var_7 = external dso_local local_unnamed_addr global i8, align 1
@var_15 = external dso_local local_unnamed_addr global i32, align 4
@var_17 = external dso_local local_unnamed_addr global i16, align 2
@arr_4 = external dso_local local_unnamed_addr global [20 x i16], align 16
@arr_11 = external dso_local local_unnamed_addr global [20 x [20 x [20 x i32]]], align 16

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: write, target_mem: none) uwtable
define dso_local noundef i32 @func() local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr @var_15, align 4, !tbaa !4
  %1 = load i64, ptr @var_0, align 8, !tbaa !8
  %conv17 = trunc i64 %1 to i8
  %conv18 = sext i8 %conv17 to i32
  %sub19 = add nsw i32 %conv18, -70
  %sext.mask = and i32 %sub19, 255
  %cmp26 = icmp eq i32 %sext.mask, 20
  tail call void @llvm.assume(i1 %cmp26)
  %tobool32.not = icmp eq i32 %0, 0
  %2 = load i16, ptr @var_17, align 2
  %.not = icmp eq i16 %2, 0
  %narrow188 = select i1 %tobool32.not, i1 true, i1 %.not
  %cmp61 = icmp slt i8 %conv17, 92
  tail call void @llvm.assume(i1 %cmp61)
  %conv95194 = sext i1 %narrow188 to i32
  %3 = load i8, ptr @var_7, align 1, !tbaa !10, !range !12, !noundef !13
  %loadedv = zext nneg i8 %3 to i64
  %sub104 = sub nsw i64 1, %1
  %cond111 = tail call i64 @llvm.smax.i64(i64 %sub104, i64 %loadedv)
  %tobool112.not = icmp eq i64 %cond111, 0
  %4 = load i32, ptr @var_3, align 4
  %5 = load i8, ptr @var_4, align 1
  %conv115 = zext i8 %5 to i32
  %cond117 = select i1 %tobool112.not, i32 %conv115, i32 %4
  %conv121 = add i32 %cond117, 65325
  %6 = and i32 %conv121, 65535
  %7 = icmp samesign ult i32 %6, 3
  %tobool136.not.us.not = icmp eq i8 %3, 0
  br i1 %7, label %for.body.lr.ph.split.us, label %for.body

for.body.lr.ph.split.us:                          ; preds = %entry
  %tobool128.not = icmp eq i32 %4, 0
  br i1 %tobool128.not, label %for.body.lr.ph.split.us.split.us, label %for.body.us.preheader

for.body.us.preheader:                            ; preds = %for.body.lr.ph.split.us
  %8 = sext i1 %narrow188 to i64
  %sext = zext nneg i32 %sub19 to i64
  br label %for.body.us

for.body.lr.ph.split.us.split.us:                 ; preds = %for.body.lr.ph.split.us
  br i1 %tobool136.not.us.not, label %for.body.us.us, label %for.body.us.us.us.preheader

for.body.us.us.us.preheader:                      ; preds = %for.body.lr.ph.split.us.split.us
  %9 = sext i1 %narrow188 to i64
  %sext215 = zext nneg i32 %sub19 to i64
  br label %for.body.us.us.us

for.body.us.us.us:                                ; preds = %for.body.us.us.us, %for.body.us.us.us.preheader
  %indvars.iv213 = phi i64 [ %9, %for.body.us.us.us.preheader ], [ %indvars.iv.next214, %for.body.us.us.us ]
  %arrayidx148.us.us.us = getelementptr inbounds [1600 x i8], ptr @arr_11, i64 %indvars.iv213
  %arrayidx150.us.us.us = getelementptr inbounds [80 x i8], ptr %arrayidx148.us.us.us, i64 %indvars.iv213
  store i32 0, ptr %arrayidx150.us.us.us, align 16, !tbaa !4
  %indvars.iv.next214 = add nsw i64 %indvars.iv213, 3
  %10 = icmp slt i64 %indvars.iv.next214, %sext215
  br i1 %10, label %for.body.us.us.us, label %for.cond.cleanup

for.body.us.us:                                   ; preds = %for.body.us.us, %for.body.lr.ph.split.us.split.us
  %indvars.iv217 = phi i32 [ %indvars.iv.next218, %for.body.us.us ], [ %conv95194, %for.body.lr.ph.split.us.split.us ]
  %indvars.iv.next218 = add nsw i32 %indvars.iv217, 3
  %cmp99.us.us = icmp sgt i32 %sub19, %indvars.iv.next218
  br i1 %cmp99.us.us, label %for.body.us.us, label %for.cond.cleanup

for.body.us:                                      ; preds = %for.cond122.for.cond.cleanup126_crit_edge.split.us204, %for.body.us.preheader
  %indvars.iv210 = phi i64 [ %8, %for.body.us.preheader ], [ %indvars.iv.next211, %for.cond122.for.cond.cleanup126_crit_edge.split.us204 ]
  %arrayidx.us = getelementptr inbounds [2 x i8], ptr @arr_4, i64 %indvars.iv210
  %11 = load i16, ptr %arrayidx.us, align 2, !tbaa !14
  %tobool136.not.us200.not = icmp eq i16 %11, 0
  br i1 %tobool136.not.us200.not, label %for.cond122.for.cond.cleanup126_crit_edge.split.us204, label %for.body127.lr.ph.split.split.us.us

for.cond122.for.cond.cleanup126_crit_edge.split.us204: ; preds = %for.body127.lr.ph.split.split.us.us, %for.body.us
  %indvars.iv.next211 = add nsw i64 %indvars.iv210, 3
  %12 = icmp slt i64 %indvars.iv.next211, %sext
  br i1 %12, label %for.body.us, label %for.cond.cleanup

for.body127.lr.ph.split.split.us.us:              ; preds = %for.body.us
  %arrayidx148.us = getelementptr inbounds [1600 x i8], ptr @arr_11, i64 %indvars.iv210
  %arrayidx150.us = getelementptr inbounds [80 x i8], ptr %arrayidx148.us, i64 %indvars.iv210
  store i32 0, ptr %arrayidx150.us, align 16, !tbaa !4
  br label %for.cond122.for.cond.cleanup126_crit_edge.split.us204

for.cond.cleanup:                                 ; preds = %for.body, %for.cond122.for.cond.cleanup126_crit_edge.split.us204, %for.body.us.us, %for.body.us.us.us
  ret i32 0

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ %conv95194, %entry ]
  %indvars.iv.next = add nsw i32 %indvars.iv, 3
  %cmp99 = icmp sgt i32 %sub19, %indvars.iv.next
  br i1 %cmp99, label %for.body, label %for.cond.cleanup
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smax.i64(i64, i64) #2

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: write, target_mem: none) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!llvm.errno.tbaa = !{!4}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 7, !"PIE Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{!"clang version 23.0.0git (/home/meinersbur/src/llvm/polly/_src/clang 458f1aae8d2da5bf786ba53ea200e94a918ff55a)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"long long", !6, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"_Bool", !6, i64 0}
!12 = !{i8 0, i8 2}
!13 = !{}
!14 = !{!15, !15, i64 0}
!15 = !{!"short", !6, i64 0}
