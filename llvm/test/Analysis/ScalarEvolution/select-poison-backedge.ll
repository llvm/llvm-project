; RUN: opt -passes='print<scalar-evolution>' -disable-output %s 2>&1 | FileCheck %s

; Some frontends emit single-block loops where loop-carried values flow
; through a select guarded by the exit condition.  SCEV should look through
; the select on the back-edge to recover the underlying AddRec.

; Test 1: select with poison
define i64 @select_poison(i64 %n, i64 %start, i64 %stop, i64 %step) {
entry:
  %empty = icmp slt i64 %n, 1
  br i1 %empty, label %exit, label %first

first:
  %first_sum = add i64 %start, 1
  %first_done = icmp eq i64 %n, 1
  %first_val_done = icmp eq i64 %start, %stop
  %first_exit = or i1 %first_done, %first_val_done
  %first_val_next = add i64 %start, %step
  br i1 %first_exit, label %exit, label %body

body:
  %accum = phi i64 [ %first_sum, %first ], [ %sum,     %body ]
  %val   = phi i64 [ %first_val_next, %first ], [ %sel.val, %body ]
  %i     = phi i64 [ 2,         %first ], [ %i.next,  %body ]
  %ab    = add i64 %val, %i
  %sum   = add i64 %ab, %accum
  %done1 = icmp eq i64 %i, %n
  %i.next = add i64 %i, 1
  %done2 = icmp eq i64 %val, %stop
  %either = select i1 %done1, i1 true, i1 %done2
  %val.next = add i64 %val, %step
  %sel.val = select i1 %done1, i64 poison, i64 %val.next
  br i1 %either, label %exit, label %body

exit:
  %result = phi i64 [ 0, %entry ], [ %first_sum, %first ], [ %sum, %body ]
  ret i64 %result
}

; CHECK-LABEL: Classifying expressions for: @select_poison
; CHECK: %val = phi
; CHECK-NEXT: -->  {(%start + %step),+,%step}<%body>

; Test 2: select with old value
define i64 @select_oldval(i64 %n, i64 %start, i64 %stop, i64 %step) {
entry:
  %guard = icmp slt i64 %n, 1
  %stop_guard = icmp eq i64 %start, %stop
  %no_entry = or i1 %guard, %stop_guard
  br i1 %no_entry, label %exit, label %body

body:
  %counter = phi i64 [ 1, %entry ], [ %sel.counter, %body ]
  %stepper = phi i64 [ %start, %entry ], [ %step.next, %body ]
  %accum   = phi i64 [ 0, %entry ], [ %sum, %body ]
  %step.next = add i64 %stepper, %step
  %ab = add i64 %counter, %step.next
  %sum = add i64 %ab, %accum
  %done1 = icmp sge i64 %counter, %n
  %counter.next = add i64 %counter, 1
  %sel.counter = select i1 %done1, i64 %counter, i64 %counter.next
  %done2 = icmp eq i64 %step.next, %stop
  %either = select i1 %done1, i1 true, i1 %done2
  br i1 %either, label %exit, label %body

exit:
  %result = phi i64 [ 0, %entry ], [ %sum, %body ]
  ret i64 %result
}

; CHECK-LABEL: Classifying expressions for: @select_oldval
; CHECK: %counter = phi
; CHECK-NEXT: -->  {1,+,1}<nuw><%body>
