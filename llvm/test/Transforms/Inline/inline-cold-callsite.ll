
; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -inline-threshold=100 -inline-cold-callsite-threshold=0 -S | FileCheck %s

; This tests that a cold callsite gets the inline-cold-callsite-threshold
; and does not get inlined. Another callsite to an identical callee that
; is not cold gets inlined because cost is below the inline-threshold.

define void @callee() "function-inline-cost"="10" {
  call void @extern()
  ret void
}

declare void @extern()
declare i1 @ext(i32)

; CHECK-LABEL: caller
define i32 @caller(i32 %n) {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret i32 0

for.body:
  %i.05 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
; CHECK: %call = tail call
  %call = tail call zeroext i1 @ext(i32 %i.05)
; CHECK-NOT: call void @callee
; CHECK-NEXT: call void @extern
  call void @callee()
  br i1 %call, label %cold, label %for.inc, !prof !0

cold:
; CHECK: call void @callee
  call void @callee()
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}


!0 = !{!"branch_weights", i32 1, i32 2000}
