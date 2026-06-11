; RUN: opt -passes=loop-distribute -enable-loop-distribute -verify-loop-info -verify-dom-info -S < %s | FileCheck %s

; LoopDistribute can call LoopVersioning on a non-LCSSA loop.
; Here %sum_add already has an exit-block LCSSA PHI, but is also used raw in
; the exit block. LoopDistribute now forms LCSSA before versioning, rewriting the raw use
; through the exit PHI, preventing the dominance issue in the cloned path.

define void @non_lcssa_exit_use(ptr %a, ptr %b, ptr %c, ptr %sum.out) {
; CHECK-LABEL: define void @non_lcssa_exit_use(
; CHECK:       for.body.lver.check:
; CHECK:       for.end:
; CHECK:         %[[LCSSA:.*]] = phi i32 [ %sum_add
; CHECK:         store i32 %[[LCSSA]], ptr %sum.out
; CHECK-NOT:     store i32 %sum_add, ptr %sum.out
entry:
  br label %for.body

for.body:
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum_add, %for.body ]
  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %ind
  %loadA = load i32, ptr %arrayidxA, align 4
  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %ind
  %loadB = load i32, ptr %arrayidxB, align 4
  %mulA = mul i32 %loadB, %loadA
  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, ptr %a, i64 %add
  store i32 %mulA, ptr %arrayidxA_plus_4, align 4
  %arrayidxC = getelementptr inbounds i32, ptr %c, i64 %ind
  %loadC = load i32, ptr %arrayidxC, align 4
  %sum_add = add nuw nsw i32 %sum, %loadC
  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  %sum_add.lcssa = phi i32 [ %sum_add, %for.body ]
  store i32 %sum_add, ptr %sum.out, align 4
  ret void
}
