; RUN: opt -S -jump-threading < %s | FileCheck %s


; CHECK-LABEL: @f(
; CHECK-LABEL: entry:
; CHECK-NEXT:  br label %[[loop:.*]]
; CHECK:       [[loop]]:
; CHECK-NEXT:  br label %[[loop]]

define void @f() {
entry:
  br label %for.cond1

if.end16:
  %phi1 = phi i32 [ undef, %for.cond1 ]
  %g.3 = phi i32 [ %g.1, %for.cond1 ]
  %sext = shl i32 %g.3, 16
  %conv20 = ashr exact i32 %sext, 16
  %tobool21 = icmp eq i32 %phi1, 0
  br i1 %tobool21, label %lor.rhs, label %for.cond1

for.cond1:
  %g.1 = phi i32 [ 0, %entry ], [ 0, %lor.rhs ], [ %g.3, %if.end16 ]
  br i1 undef, label %lor.rhs, label %if.end16

lor.rhs:
  br label %for.cond1
}
