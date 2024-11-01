; RUN: opt -S -passes=jump-threading < %s | FileCheck %s

; CHECK-LABEL: @f(
; CHECK-LABEL: entry
; CHECK-NEXT: ret void
;
; JumpThreading must detect the next two blocks are unreachable from entry
; and leave them alone. A subsequent pass will remove them from @f.
;
; CHECK: for.cond1:
; CHECK-NEXT: phi
; CHECK-NEXT: icmp
; CHECK-NEXT: br i1 %cmp, label %for.body, label %for.cond1
; CHECK: for.body:
; CHECK-NEXT: add
; CHECK-NEXT: icmp
; CHECK-NEXT: br i1 %a, label %for.cond1, label %for.cond1

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
