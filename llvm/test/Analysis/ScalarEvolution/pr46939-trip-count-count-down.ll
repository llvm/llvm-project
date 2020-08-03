; RUN: opt -analyze -scalar-evolution %s 2>&1 | FileCheck %s

declare void @iteration()

define void @reverse_loop(i32 %n) {
; CHECK-LABEL: 'reverse_loop'
; CHECK-NEXT:  Classifying expressions for: @reverse_loop
; CHECK-NEXT:    %i.011 = phi i32 [ %n, %for.body.lr.ph ], [ %dec, %for.body ]
; CHECK-NEXT:    --> {%n,+,-1}<nsw><%for.body> U: full-set S: full-set Exits: 0 LoopDispositions: { %for.body: Computable }
; CHECK-NEXT:    %dec = add nsw i32 %i.011, -1
; CHECK-NEXT:    --> {(-1 + %n),+,-1}<nw><%for.body> U: full-set S: full-set Exits: -1 LoopDispositions: { %for.body: Computable }
; CHECK-NEXT:  Determining loop execution counts for: @reverse_loop
; CHECK-NEXT:  Loop %for.body: backedge-taken count is %n
; CHECK-NEXT:  Loop %for.body: max backedge-taken count is 2147483647
; CHECK-NEXT:  Loop %for.body: Predicated backedge-taken count is %n
; CHECK-NEXT:   Predicates:
; CHECK:       Loop %for.body: Trip multiple is 1
;
entry:
  %cmp10 = icmp sgt i32 %n, -1
  br i1 %cmp10, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:
  br label %for.body

for.body:
  %i.011 = phi i32 [ %n, %for.body.lr.ph ], [ %dec, %for.body ]
  call void @iteration()
  %dec = add nsw i32 %i.011, -1
  %cmp = icmp sgt i32 %i.011, 0
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void
}
