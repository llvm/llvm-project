; This test verifies that the loop vectorizer will NOT produce a tail
; loop with the optimize for size or the minimize size attributes.
; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -pgso -S | FileCheck %s -check-prefix=PGSO
; RUN: opt < %s -loop-vectorize -pgso=false -S | FileCheck %s -check-prefix=NPGSO

target datalayout = "E-m:e-p:32:32-i64:32-f64:32:64-a:0:32-n32-S128"

@tab = common global [32 x i8] zeroinitializer, align 1

define i32 @foo_optsize() #0 {
; CHECK-LABEL: @foo_optsize(
; CHECK-NOT: <2 x i8>
; CHECK-NOT: <4 x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %i.08, 202
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

attributes #0 = { optsize }

define i32 @foo_minsize() #1 {
; CHECK-LABEL: @foo_minsize(
; CHECK-NOT: <2 x i8>
; CHECK-NOT: <4 x i8>
; CHECK-LABEL: @foo_pgso(

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %i.08, 202
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

attributes #1 = { minsize }

define i32 @foo_pgso() !prof !14 {
; PGSO-LABEL: @foo_pgso(
; PGSO-NOT: <{{[0-9]+}} x i8>
; NPGSO-LABEL: @foo_pgso(
; NPGSO: <{{[0-9]+}} x i8>

entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %i.08, 202
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

; PR43371: don't run into an assert due to emitting SCEV runtime checks
; with OptForSize.
;
@cm_array = external global [2592 x i16], align 1

define void @pr43371() optsize {
;
; CHECK-LABEL: @pr43371
; CHECK-NOT:   vector.scevcheck
;
; We do not want to generate SCEV predicates when optimising for size, because
; that will lead to extra code generation such as the SCEV overflow runtime
; checks. Not generating SCEV predicates can still result in vectorisation as
; the non-consecutive loads/stores can be scalarized:
;
; CHECK: vector.body:
; CHECK: store i16 0, i16* %{{.*}}, align 1
; CHECK: store i16 0, i16* %{{.*}}, align 1
; CHECK: br i1 {{.*}}, label %vector.body
;
entry:
  br label %for.body29

for.cond.cleanup28:
  unreachable

for.body29:
  %i24.0170 = phi i16 [ 0, %entry], [ %inc37, %for.body29]
  %add33 = add i16 undef, %i24.0170
  %idxprom34 = zext i16 %add33 to i32
  %arrayidx35 = getelementptr [2592 x i16], [2592 x i16] * @cm_array, i32 0, i32 %idxprom34
  store i16 0, i16 * %arrayidx35, align 1
  %inc37 = add i16 %i24.0170, 1
  %cmp26 = icmp ult i16 %inc37, 756
  br i1 %cmp26, label %for.body29, label %for.cond.cleanup28
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 0}
