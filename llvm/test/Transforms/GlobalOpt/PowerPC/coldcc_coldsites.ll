; RUN: opt -passes=globalopt -mtriple=powerpc64le-unknown-linux-gnu -ppc-enable-coldcc -S < %s | FileCheck %s -check-prefix=COLDCC
; RUN: opt -passes=globalopt -S < %s | FileCheck %s -check-prefix=CHECK

define signext i32 @caller(i32 signext %a, i32 signext %b, i32 signext %lim, i32 signext %i) local_unnamed_addr #0 !prof !30 {
entry:
; COLDCC: call coldcc signext i32 @callee
; CHECK: call fastcc signext i32 @callee
  %add = add nsw i32 %b, %a
  %sub = add nsw i32 %lim, -1
  %cmp = icmp eq i32 %sub, %i
  br i1 %cmp, label %if.then, label %if.end, !prof !31

if.then:                                          ; preds = %entry
  %call = tail call signext i32 @callee(i32 signext %a, i32 signext %b)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %f.0 = phi i32 [ %call, %if.then ], [ %add, %entry ]
  ret i32 %f.0
}

define internal signext i32 @callee(i32 signext %a, i32 signext %b) unnamed_addr #0 {
entry:
  %0 = tail call i32 asm "add $0, $1, $2", "=r,r,r,~{r6},~{r7},~{r8},~{r9}"(i32 %a, i32 %b) #1, !srcloc !32
  %mul = mul nsw i32 %a, 3
  %mul1 = shl i32 %0, 1
  %add = add nsw i32 %mul1, %mul
  ret i32 %add
}

define signext i32 @main() local_unnamed_addr #0 !prof !33 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

for.body:                                         ; preds = %for.body, %entry
  %i.011 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %ret.010 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %call = tail call signext i32 @caller(i32 signext 4, i32 signext 5, i32 signext 10000000, i32 signext %i.011)
  %add = add nsw i32 %call, %ret.010
  %inc = add nuw nsw i32 %i.011, 1
  %exitcond = icmp eq i32 %inc, 10000000
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !prof !34
}
attributes #0 = { noinline }

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 20000003}
!4 = !{!"MaxCount", i64 10000000}
!5 = !{!"MaxInternalCount", i64 10000000}
!6 = !{!"MaxFunctionCount", i64 10000000}
!7 = !{!"NumCounts", i64 5}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13, !14, !15, !16, !16, !17, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26}
!11 = !{i32 10000, i64 10000000, i32 2}
!12 = !{i32 100000, i64 10000000, i32 2}
!13 = !{i32 200000, i64 10000000, i32 2}
!14 = !{i32 300000, i64 10000000, i32 2}
!15 = !{i32 400000, i64 10000000, i32 2}
!16 = !{i32 500000, i64 10000000, i32 2}
!17 = !{i32 600000, i64 10000000, i32 2}
!18 = !{i32 700000, i64 10000000, i32 2}
!19 = !{i32 800000, i64 10000000, i32 2}
!20 = !{i32 900000, i64 10000000, i32 2}
!21 = !{i32 950000, i64 10000000, i32 2}
!22 = !{i32 990000, i64 10000000, i32 2}
!23 = !{i32 999000, i64 10000000, i32 2}
!24 = !{i32 999900, i64 10000000, i32 2}
!25 = !{i32 999990, i64 10000000, i32 2}
!26 = !{i32 999999, i64 10000000, i32 2}
!30 = !{!"function_entry_count", i64 10000000}
!31 = !{!"branch_weights", i32 2, i32 10000000}
!32 = !{i32 59}
!33 = !{!"function_entry_count", i64 1}
!34 = !{!"branch_weights", i32 2, i32 10000001}
