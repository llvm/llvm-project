; REQUIRES: asserts
; RUN: opt < %s -S -debug-only=loop-unroll -passes=loop-unroll -unroll-runtime 2>&1 | FileCheck %s
; RUN: opt < %s -S -debug-only=loop-unroll -passes='require<profile-summary>,function(require<opt-remark-emit>,loop-unroll)' 2>&1 | FileCheck %s
; RUN: opt < %s -S -debug-only=loop-unroll -passes='require<profile-summary>,function(require<opt-remark-emit>,loop-unroll<no-profile-peeling>)' 2>&1 | FileCheck %s --check-prefixes=CHECK-NO-PEEL

; Make sure we use the profile information correctly to peel-off 3 iterations
; from the loop, and update the branch weights for the peeled loop properly.
; All side exits to deopt does not change weigths.

; CHECK: Loop Unroll: F[basic]
; CHECK: PEELING loop %for.body with iteration count 4!
; CHECK-NO-PEEL-NOT: PEELING loop %for.body
; CHECK-LABEL: @basic
; CHECK: br i1 %c, label %{{.*}}, label %side_exit, !prof !15
; CHECK: br i1 %{{.*}}, label %[[NEXT0:.*]], label %for.cond.for.end_crit_edge, !prof !16
; CHECK: [[NEXT0]]:
; CHECK: br i1 %c, label %{{.*}}, label %side_exit, !prof !15
; CHECK: br i1 %{{.*}}, label %[[NEXT1:.*]], label %for.cond.for.end_crit_edge, !prof !17
; CHECK: [[NEXT1]]:
; CHECK: br i1 %c, label %{{.*}}, label %side_exit, !prof !15
; CHECK: br i1 %{{.*}}, label %[[NEXT2:.*]], label %for.cond.for.end_crit_edge, !prof !18
; CHECK: [[NEXT2]]:
; CHECK: br i1 %c, label %{{.*}}, label %side_exit.loopexit, !prof !15
; CHECK: br i1 %{{.*}}, label %for.body, label %{{.*}}, !prof !18

define i32 @basic(ptr %p, i32 %k, i1 %c) #0 !prof !15 {
entry:
  %cmp3 = icmp slt i32 0, %k
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.05 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %continue ]
  %p.addr.04 = phi ptr [ %p, %for.body.lr.ph ], [ %incdec.ptr, %continue ]
  %incdec.ptr = getelementptr inbounds i32, ptr %p.addr.04, i32 1
  store i32 %i.05, ptr %p.addr.04, align 4
  %inc = add nsw i32 %i.05, 1
  %cmp = icmp slt i32 %inc, %k
  br i1 %c, label %continue, label %side_exit, !prof !17

continue:
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge, !prof !16

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %res = phi i32 [ 0, %entry ], [ %inc, %for.cond.for.end_crit_edge ]
  ret i32 %res

side_exit:
  %rval = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %inc) ]
  ret i32 %rval
}

declare i32 @llvm.experimental.deoptimize.i32(...)

attributes #0 = { nounwind }
attributes #1 = { nounwind optsize }

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10}
!5 = !{!"MaxCount", i64 3}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 3}
!8 = !{!"NumCounts", i64 2}
!9 = !{!"NumFunctions", i64 2}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 3, i32 2}
!13 = !{i32 999000, i64 1, i32 10}
!14 = !{i32 999999, i64 1, i32 10}
!15 = !{!"function_entry_count", i64 1}
!16 = !{!"branch_weights", i32 3001, i32 1001}
!17 = !{!"branch_weights", i32 1, i32 0}

; This is a weights of deopt side-exit.
;CHECK: !15 = !{!"branch_weights", i32 1, i32 0}
; This is a weights of latch and its copies.
;CHECK: !16 = !{!"branch_weights", i32 3001, i32 1001}
;CHECK: !17 = !{!"branch_weights", i32 2000, i32 1001}
;CHECK: !18 = !{!"branch_weights", i32 1001, i32 1001}

