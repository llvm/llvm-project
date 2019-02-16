; REQUIRES: asserts
; RUN: opt < %s -loop-unswitch -disable-output -stats 2>&1| FileCheck %s
; RUN: opt < %s -loop-unswitch -enable-mssa-loop-dependency=true -verify-memoryssa -disable-output -stats 2>&1| FileCheck %s

; Check the select statement in the loop will be unswitched.
; CHECK: 1 loop-unswitch - Number of selects unswitched
define i32 @test(i1 zeroext %x, i32 %a) local_unnamed_addr #0 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %s.0 = phi i32 [ %a, %entry ], [ %add, %while.body ]
  %cmp = icmp slt i32 %i.0, 10000
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %cond = select i1 %x, i32 %a, i32 %i.0
  %add = add nsw i32 %s.0, %cond
  %inc = add nsw i32 %i.0, 1
  br label %while.cond

while.end:                                        ; preds = %while.cond
  %s.0.lcssa = phi i32 [ %s.0, %while.cond ]
  ret i32 %s.0.lcssa
}

