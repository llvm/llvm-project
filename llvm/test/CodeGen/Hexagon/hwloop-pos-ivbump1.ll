;RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that a hardware loop is not generaetd due to a potential
; underflow.

; CHECK-NOT: loop0

define i32 @main() #0 {
entry:
  br label %while.cond.outer

while.cond.outer.loopexit:
  %.lcssa = phi i32 [ %0, %for.body.preheader ]
  br label %while.cond.outer

while.cond.outer:
  %i.0.ph = phi i32 [ 0, %entry ], [ 3, %while.cond.outer.loopexit ]
  %j.0.ph = phi i32 [ 0, %entry ], [ %.lcssa, %while.cond.outer.loopexit ]
  %k.0.ph = phi i32 [ 0, %entry ], [ 1, %while.cond.outer.loopexit ]
  br label %while.cond

while.cond:
  %i.0 = phi i32 [ %i.0.ph, %while.cond.outer ], [ %inc, %for.body.preheader ]
  %j.0 = phi i32 [ %j.0.ph, %while.cond.outer ], [ %0, %for.body.preheader ]
  %inc = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %i.0, 4
  br i1 %cmp, label %for.body.preheader, label %while.end

for.body.preheader:
  %0 = add i32 %j.0, 3
  %cmp5 = icmp eq i32 %inc, 3
  br i1 %cmp5, label %while.cond.outer.loopexit, label %while.cond

while.end:
  %k.0.ph.lcssa = phi i32 [ %k.0.ph, %while.cond ]
  %inc.lcssa = phi i32 [ %inc, %while.cond ]
  %j.0.lcssa = phi i32 [ %j.0, %while.cond ]
  %cmp6 = icmp ne i32 %inc.lcssa, 5
  %cmp7 = icmp ne i32 %j.0.lcssa, 12
  %or.cond = or i1 %cmp6, %cmp7
  %cmp9 = icmp ne i32 %k.0.ph.lcssa, 1
  %or.cond12 = or i1 %or.cond, %cmp9
  %locflg.0 = zext i1 %or.cond12 to i32
  ret i32 %locflg.0
}
