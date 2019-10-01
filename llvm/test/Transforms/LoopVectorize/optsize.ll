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
