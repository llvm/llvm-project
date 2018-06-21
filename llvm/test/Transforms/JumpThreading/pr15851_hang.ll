; RUN: opt -S -jump-threading < %s | FileCheck %s

; CHECK-LABEL: @f(
; CHECK-LABEL: entry
; CHECK: ret void
; CHECK-NOT: for.cond1
; CHECK-NOT: for.body

define void @f() {
entry:
  ret void

for.cond1:
  %i.025 = phi i32 [ %inc, %for.body ], [ %inc, %for.body ], [ 1, %for.cond1 ]
  %cmp = icmp slt i32 %i.025, 2
  br i1 %cmp, label %for.body, label %for.cond1

for.body:
  %inc = add nsw i32 %i.025, 0
  %a = icmp ugt i32 %inc, 2
  br i1 %a, label %for.cond1, label %for.cond1
}
