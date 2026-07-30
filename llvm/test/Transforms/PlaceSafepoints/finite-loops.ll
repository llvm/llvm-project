; Tests to ensure that we are not placing backedge safepoints in
; loops which are clearly finite.
;; RUN: opt < %s -passes=place-safepoints -spp-counted-loop-trip-width=32 -S | FileCheck %s
;; RUN: opt < %s -passes=place-safepoints -spp-counted-loop-trip-width=64 -S | FileCheck %s -check-prefix=COUNTED-64


; A simple counted loop with trivially known range
define void @test1(i32) gc "statepoint-example" {
; CHECK-LABEL: test1
; CHECK-LABEL: entry
; CHECK: call void @do_safepoint
; CHECK-LABEL: loop
; CHECK-NOT: call void @do_safepoint
; CHECK-LABEL: exit

entry:
  br label %loop

loop:
  %counter = phi i32 [ 0 , %entry ], [ %counter.inc , %loop ]
  %counter.inc = add i32 %counter, 1
  %counter.cmp = icmp slt i32 %counter.inc, 16
  br i1 %counter.cmp, label %loop, label %exit

exit:
  ret void
}

; The same counted loop, but with an unknown early exit
define void @test2(i1 %c, i32) gc "statepoint-example" {
; CHECK-LABEL: test2
; CHECK-LABEL: entry
; CHECK: call void @do_safepoint
; CHECK-LABEL: loop
; CHECK-NOT: call void @do_safepoint
; CHECK-LABEL: exit

entry:
  br label %loop

loop:
  %counter = phi i32 [ 0 , %entry ], [ %counter.inc , %continue ]
  %counter.inc = add i32 %counter, 1
  %counter.cmp = icmp slt i32 %counter.inc, 16
  br i1 %c, label %continue, label %exit

continue:
  br i1 %counter.cmp, label %loop, label %exit

exit:
  ret void
}

; The range is a 8 bit value and we can't overflow
define void @test3(i8 %upper) gc "statepoint-example" {
; CHECK-LABEL: test3
; CHECK-LABEL: entry
; CHECK: call void @do_safepoint
; CHECK-LABEL: loop
; CHECK-NOT: call void @do_safepoint
; CHECK-LABEL: exit

entry:
  br label %loop

loop:
  %counter = phi i8 [ 0 , %entry ], [ %counter.inc , %loop ]
  %counter.inc = add nsw i8 %counter, 1
  %counter.cmp = icmp slt i8 %counter.inc, %upper
  br i1 %counter.cmp, label %loop, label %exit

exit:
  ret void
}

; The range is a 64 bit value
define void @test4(i64 %upper) gc "statepoint-example" {
; CHECK-LABEL: test4
; CHECK-LABEL: entry
; CHECK: call void @do_safepoint
; CHECK-LABEL: loop
; CHECK: call void @do_safepoint
; CHECK-LABEL: exit

; COUNTED-64-LABEL: test4
; COUNTED-64-LABEL: entry
; COUNTED-64: call void @do_safepoint
; COUNTED-64-LABEL: loop
; COUNTED-64-NOT: call void @do_safepoint
; COUNTED-64-LABEL: exit

entry:
  br label %loop

loop:
  %counter = phi i64 [ 0 , %entry ], [ %counter.inc , %loop ]
  %counter.inc = add i64 %counter, 1
  %counter.cmp = icmp slt i64 %counter.inc, %upper
  br i1 %counter.cmp, label %loop, label %exit

exit:
  ret void
}

; This loop can run infinitely (for %upper == INT64_MAX) so it needs a
; safepoint.
define void @test5(i64 %upper) gc "statepoint-example" {
; CHECK-LABEL: test5
; CHECK-LABEL: entry
; CHECK: call void @do_safepoint
; CHECK-LABEL: loop
; CHECK: call void @do_safepoint
; CHECK-LABEL: exit

; COUNTED-64-LABEL: test5
; COUNTED-64-LABEL: entry
; COUNTED-64: call void @do_safepoint
; COUNTED-64-LABEL: loop
; COUNTED-64: call void @do_safepoint
; COUNTED-64-LABEL: exit

entry:
  br label %loop

loop:
  %counter = phi i64 [ 0 , %entry ], [ %counter.inc , %loop ]
  %counter.inc = add i64 %counter, 1
  %counter.cmp = icmp sle i64 %counter.inc, %upper
  br i1 %counter.cmp, label %loop, label %exit

exit:
  ret void
}


; This function is inlined when inserting a poll.
declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
entry:
  call void @do_safepoint()
  ret void
}
