; RUN: opt < %s -correlated-propagation -S | FileCheck %s

; Removed several cases from switch.
define i32 @switch1(i32 %s) {
; CHECK-LABEL: @switch1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[S:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[NEGATIVE:%.*]], label [[OUT:%.*]]
;
entry:
  %cmp = icmp slt i32 %s, 0
  br i1 %cmp, label %negative, label %out

negative:
; CHECK:       negative:
; CHECK-NEXT:    switch i32 [[S]], label [[OUT]] [
; CHECK-NEXT:    i32 -2, label [[NEXT:%.*]]
; CHECK-NEXT:    i32 -1, label [[NEXT]]
  switch i32 %s, label %out [
  i32 0, label %out
  i32 1, label %out
  i32 -1, label %next
  i32 -2, label %next
  i32 2, label %out
  i32 3, label %out
; CHECK-NEXT: !prof ![[MD0:[0-9]+]]
  ], !prof !{!"branch_weights", i32 99, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6}

out:
  %p = phi i32 [ 1, %entry ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ]
  ret i32 %p

next:
  %q = phi i32 [ 0, %negative ], [ 0, %negative ]
  ret i32 %q
}

; Removed all cases from switch.
define i32 @switch2(i32 %s) {
; CHECK-LABEL: @switch2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp sgt i32 [[S:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[POSITIVE:%.*]], label [[OUT:%.*]]
;
entry:
  %cmp = icmp sgt i32 %s, 0
  br i1 %cmp, label %positive, label %out

positive:
  switch i32 %s, label %out [
  i32 0, label %out
  i32 -1, label %next
  i32 -2, label %next
  ], !prof !{!"branch_weights", i32 99, i32 1, i32 2, i32 3}

out:
  %p = phi i32 [ -1, %entry ], [ 1, %positive ], [ 1, %positive ]
  ret i32 %p

next:
  %q = phi i32 [ 0, %positive ], [ 0, %positive ]
  ret i32 %q
}

; Change switch into conditional branch.
define i32 @switch3(i32 %s) {
; CHECK-LABEL: @switch3(
;
entry:
  %cmp = icmp sgt i32 %s, 0
  br i1 %cmp, label %positive, label %out

positive:
; CHECK:      positive:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 %s, 1
; CHECK-NEXT:    br i1 [[CMP]], label [[NEXT:%.*]], label [[OUT:%.*]], !prof ![[MD1:[0-9]+]]
  switch i32 %s, label %out [
  i32 1, label %next
  i32 -1, label %next
  i32 -2, label %next
  ], !prof !{!"branch_weights", i32 99, i32 1, i32 2, i32 3}

out:
  %p = phi i32 [ -1, %entry ], [ 1, %positive ]
  ret i32 %p

next:
  %q = phi i32 [ 0, %positive ], [ 0, %positive ], [ 0, %positive ]
  ret i32 %q
}

; Removed all cases from switch.
define i32 @switch4(i32 %s) {
; CHECK-LABEL: @switch4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp slt i32 [[S:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[NEGATIVE:%.*]], label [[OUT:%.*]]
;
entry:
  %cmp = icmp slt i32 %s, 0
  br i1 %cmp, label %negative, label %out

negative:
; CHECK:       negative:
; CHECK-NEXT:    br label %out
  switch i32 %s, label %out [
  i32 0, label %out
  i32 1, label %out
  i32 2, label %out
  i32 3, label %out
  ], !prof !{!"branch_weights", i32 99, i32 1, i32 2, i32 3, i32 4}

out:
  %p = phi i32 [ 1, %entry ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ]
  ret i32 %p
}

; CHECK: ![[MD0]] = !{!"branch_weights", i32 99, i32 4, i32 3}
; CHECK: ![[MD1]] = !{!"branch_weights", i32 1, i32 99}
