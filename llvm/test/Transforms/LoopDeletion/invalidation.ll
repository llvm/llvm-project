; Ensure we don't run analyses over loops after they've been deleted. We run
; one version with a no-op loop pass to make sure that the loop doesn't get
; simplified away.
;
; RUN: opt < %s -passes='require<ivusers>,no-op-loop,require<ivusers>' -S \
; RUN:     | FileCheck %s --check-prefixes=CHECK,BEFORE
; RUN: opt < %s -passes='require<ivusers>,loop-deletion,require<ivusers>' -S \
; RUN:     | FileCheck %s --check-prefixes=CHECK,AFTER


define void @foo(i64 %n, i64 %m) nounwind {
; CHECK-LABEL: @foo(

entry:
  br label %bb
; CHECK:       entry:
; BEFORE-NEXT:   br label %bb
; AFTER-NEXT:    br label %return

bb:
  %x.0 = phi i64 [ 0, %entry ], [ %t0, %bb2 ]
  %t0 = add i64 %x.0, 1
  %t1 = icmp slt i64 %x.0, %n
  br i1 %t1, label %bb2, label %return
; BEFORE:      bb:
; BEFORE:        br i1 {{.*}}, label %bb2, label %return
; AFTER-NOT:   bb:
; AFTER-NOT:     br

bb2:
  %t2 = icmp slt i64 %x.0, %m
  br i1 %t1, label %bb, label %return
; BEFORE:      bb2:
; BEFORE:        br i1 {{.*}}, label %bb, label %return
; AFTER-NOT:   bb2:
; AFTER-NOT:     br

return:
  ret void
; CHECK:       return:
; CHECK-NEXT:    ret void
}
