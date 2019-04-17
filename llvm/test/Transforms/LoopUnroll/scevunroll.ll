; RUN: opt < %s -S -indvars -loop-unroll -verify-loop-info | FileCheck %s
;
; Unit tests for loop unrolling using ScalarEvolution to compute trip counts.
;
; Indvars is run first to generate an "old" SCEV result. Some unit
; tests may check that SCEV is properly invalidated between passes.

; Completely unroll loops without a canonical IV.
;
; CHECK-LABEL: @sansCanonical(
; CHECK-NOT: phi
; CHECK-NOT: icmp
; CHECK: ret
define i32 @sansCanonical(i32* %base) nounwind {
entry:
  br label %while.body

while.body:
  %iv = phi i64 [ 10, %entry ], [ %iv.next, %while.body ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %while.body ]
  %iv.next = add i64 %iv, -1
  %adr = getelementptr inbounds i32, i32* %base, i64 %iv.next
  %tmp = load i32, i32* %adr, align 8
  %sum.next = add i32 %sum, %tmp
  %iv.narrow = trunc i64 %iv.next to i32
  %cmp.i65 = icmp sgt i32 %iv.narrow, 0
  br i1 %cmp.i65, label %while.body, label %exit

exit:
  ret i32 %sum
}

; SCEV unrolling properly handles loops with multiple exits. In this
; case, the computed trip count based on a canonical IV is *not* for a
; latch block. Canonical unrolling incorrectly unrolls it, but SCEV
; unrolling does not.
;
; CHECK-LABEL: @earlyLoopTest(
; CHECK: tail:
; CHECK-NOT: br
; CHECK: br i1 %cmp2, label %loop, label %exit2
define i64 @earlyLoopTest(i64* %base) nounwind {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %inc, %tail ]
  %s = phi i64 [ 0, %entry ], [ %s.next, %tail ]
  %adr = getelementptr i64, i64* %base, i64 %iv
  %val = load i64, i64* %adr
  %s.next = add i64 %s, %val
  %inc = add i64 %iv, 1
  %cmp = icmp ne i64 %inc, 4
  br i1 %cmp, label %tail, label %exit1

tail:
  %cmp2 = icmp ne i64 %val, 0
  br i1 %cmp2, label %loop, label %exit2

exit1:
  ret i64 %s

exit2:
  ret i64 %s.next
}

; SCEV properly unrolls multi-exit loops.
;
; CHECK-LABEL: @multiExit(
; CHECK: getelementptr i32, i32* %base, i32 10
; CHECK-NEXT: load i32, i32*
; CHECK: br i1 false, label %l2.10, label %exit1
; CHECK: l2.10:
; CHECK-NOT: br
; CHECK: ret i32
define i32 @multiExit(i32* %base) nounwind {
entry:
  br label %l1
l1:
  %iv1 = phi i32 [ 0, %entry ], [ %inc1, %l2 ]
  %iv2 = phi i32 [ 0, %entry ], [ %inc2, %l2 ]
  %inc1 = add i32 %iv1, 1
  %inc2 = add i32 %iv2, 1
  %adr = getelementptr i32, i32* %base, i32 %iv1
  %val = load i32, i32* %adr
  %cmp1 = icmp slt i32 %iv1, 5
  br i1 %cmp1, label %l2, label %exit1
l2:
  %cmp2 = icmp slt i32 %iv2, 10
  br i1 %cmp2, label %l1, label %exit2
exit1:
  ret i32 1
exit2:
  ret i32 %val
}


; SCEV should not unroll a multi-exit loops unless the latch block has
; a known trip count, regardless of the early exit trip counts. The
; LoopUnroll utility uses this assumption to optimize the latch
; block's branch.
;
; CHECK-LABEL: @multiExitIncomplete(
; CHECK: l3:
; CHECK-NOT: br
; CHECK:   br i1 %cmp3, label %l1, label %exit3
define i32 @multiExitIncomplete(i32* %base) nounwind {
entry:
  br label %l1
l1:
  %iv1 = phi i32 [ 0, %entry ], [ %inc1, %l3 ]
  %iv2 = phi i32 [ 0, %entry ], [ %inc2, %l3 ]
  %inc1 = add i32 %iv1, 1
  %inc2 = add i32 %iv2, 1
  %adr = getelementptr i32, i32* %base, i32 %iv1
  %val = load i32, i32* %adr
  %cmp1 = icmp slt i32 %iv1, 5
  br i1 %cmp1, label %l2, label %exit1
l2:
  %cmp2 = icmp slt i32 %iv2, 10
  br i1 %cmp2, label %l3, label %exit2
l3:
  %cmp3 = icmp ne i32 %val, 0
  br i1 %cmp3, label %l1, label %exit3

exit1:
  ret i32 1
exit2:
  ret i32 2
exit3:
  ret i32 3
}

; When loop unroll merges a loop exit with one of its parent loop's
; exits, SCEV must forget its ExitNotTaken info.
;
; CHECK-LABEL: @nestedUnroll(
; CHECK-NOT: br i1
; CHECK: for.body87:
define void @nestedUnroll() nounwind {
entry:
  br label %for.inc

for.inc:
  br i1 false, label %for.inc, label %for.body38.preheader

for.body38.preheader:
  br label %for.body38

for.body38:
  %i.113 = phi i32 [ %inc76, %for.inc74 ], [ 0, %for.body38.preheader ]
  %mul48 = mul nsw i32 %i.113, 6
  br label %for.body43

for.body43:
  %j.011 = phi i32 [ 0, %for.body38 ], [ %inc72, %for.body43 ]
  %add49 = add nsw i32 %j.011, %mul48
  %sh_prom50 = zext i32 %add49 to i64
  %inc72 = add nsw i32 %j.011, 1
  br i1 false, label %for.body43, label %for.inc74

for.inc74:
  %inc76 = add nsw i32 %i.113, 1
  br i1 false, label %for.body38, label %for.body87.preheader

for.body87.preheader:
  br label %for.body87

for.body87:
  br label %for.body87
}

; PR16130: clang produces incorrect code with loop/expression at -O2
; rdar:14036816 loop-unroll makes assumptions about undefined behavior
;
; The loop latch is assumed to exit after the first iteration because
; of the induction variable's NSW flag. However, the loop latch's
; equality test is skipped and the loop exits after the second
; iteration via the early exit. So loop unrolling cannot assume that
; the loop latch's exit count of zero is an upper bound on the number
; of iterations.
;
; CHECK-LABEL: @nsw_latch(
; CHECK: for.body:
; CHECK: %b.03 = phi i32 [ 0, %entry ], [ %add, %for.cond ]
; CHECK: return:
; CHECK: %b.03.lcssa = phi i32 [ %b.03, %for.body ], [ %b.03, %for.cond ]
define void @nsw_latch(i32* %a) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %for.cond, %entry
  %b.03 = phi i32 [ 0, %entry ], [ %add, %for.cond ]
  %tobool = icmp eq i32 %b.03, 0
  %add = add nsw i32 %b.03, 8
  br i1 %tobool, label %for.cond, label %return

for.cond:                                         ; preds = %for.body
  %cmp = icmp eq i32 %add, 13
  br i1 %cmp, label %return, label %for.body

return:                                           ; preds = %for.body, %for.cond
  %b.03.lcssa = phi i32 [ %b.03, %for.body ], [ %b.03, %for.cond ]
  %retval.0 = phi i32 [ 1, %for.body ], [ 0, %for.cond ]
  store i32 %b.03.lcssa, i32* %a, align 4
  ret void
}
