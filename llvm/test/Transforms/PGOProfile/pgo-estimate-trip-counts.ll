; Check the pgo-estimate-trip-counts pass.  Indirectly check
; llvm::getLoopEstimatedTripCount and llvm::setLoopEstimatedTripCount.

; RUN: opt %s -S -passes=pgo-estimate-trip-counts 2>&1 | \
; RUN:   FileCheck %s -implicit-check-not='{{^[^ ;]*:}}'

; No metadata and trip count is estimable: create metadata with value.
;
; CHECK-LABEL: define void @estimable(i32 %n) {
define void @estimable(i32 %n) {
; CHECK: entry:
entry:
  br label %body

; CHECK: body:
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  ; CHECK: br i1 %cmp, label %body, label %exit, !prof !0, !llvm.loop ![[#ESTIMABLE:]]
  br i1 %cmp, label %body, label %exit, !prof !0

; CHECK: exit:
exit:
  ret void
}

; No metadata and trip count is inestimable because no branch weights: create
; metadata with no value.
;
; CHECK-LABEL: define void @no_branch_weights(i32 %n) {
define void @no_branch_weights(i32 %n) {
; CHECK: entry:
entry:
  br label %body

; CHECK: body:
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  ; CHECK: br i1 %cmp, label %body, label %exit, !llvm.loop ![[#NO_BRANCH_WEIGHTS:]]
  br i1 %cmp, label %body, label %exit

; CHECK: exit:
exit:
  ret void
}

; No metadata and trip count is inestimable because multiple latches: create
; metadata with no value.
;
; CHECK-LABEL: define void @multi_latch(i32 %n, i1 %c) {
define void @multi_latch(i32 %n, i1 %c) {
; CHECK: entry:
entry:
  br label %head

; CHECK: head:
head:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch0], [ %inc, %latch1 ]
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  ; CHECK: br i1 %cmp, label %latch0, label %exit, !prof !0
  br i1 %cmp, label %latch0, label %exit, !prof !0

; CHECK: latch0:
latch0:
  ; CHECK: br i1 %c, label %head, label %latch1, !prof !0, !llvm.loop ![[#MULTI_LATCH:]]
  br i1 %c, label %head, label %latch1, !prof !0

; CHECK: latch1:
latch1:
  ; CHECK: br label %head
  br label %head

; CHECK: exit:
exit:
  ret void
}

; Metadata is already present with value, and trip count is estimable: keep the
; existing metadata value.
;
; CHECK-LABEL: define void @val_estimable(i32 %n) {
define void @val_estimable(i32 %n) {
; CHECK: entry:
entry:
  br label %body

; CHECK: body:
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  ; CHECK: br i1 %cmp, label %body, label %exit, !prof !0, !llvm.loop ![[#VAL_ESTIMABLE:]]
  br i1 %cmp, label %body, label %exit, !prof !0, !llvm.loop !1

; CHECK: exit:
exit:
  ret void
}

; Metadata is already present with value, and trip count is inestimable: keep
; the existing metadata value.
;
; CHECK-LABEL: define void @val_inestimable(i32 %n) {
define void @val_inestimable(i32 %n) {
; CHECK: entry:
entry:
  br label %body

; CHECK: body:
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  ; CHECK: br i1 %cmp, label %body, label %exit, !llvm.loop ![[#VAL_INESTIMABLE:]]
  br i1 %cmp, label %body, label %exit, !llvm.loop !3

; CHECK: exit:
exit:
  ret void
}

; Metadata is already present without value, and trip count is estimable: add
; new value to metadata.
;
; CHECK-LABEL: define void @no_val_estimable(i32 %n) {
define void @no_val_estimable(i32 %n) {
; CHECK: entry:
entry:
  br label %body

; CHECK: body:
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  ; CHECK: br i1 %cmp, label %body, label %exit, !prof !0, !llvm.loop ![[#NO_VAL_ESTIMABLE:]]
  br i1 %cmp, label %body, label %exit, !prof !0, !llvm.loop !5

; CHECK: exit:
exit:
  ret void
}

; Metadata is already present without value, and trip count is inestimable:
; leave no value on metadata.
;
; CHECK-LABEL: define void @no_val_inestimable(i32 %n) {
define void @no_val_inestimable(i32 %n) {
; CHECK: entry:
entry:
  br label %body

; CHECK: body:
body:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  ; CHECK: br i1 %cmp, label %body, label %exit, !llvm.loop ![[#NO_VAL_INESTIMABLE:]]
  br i1 %cmp, label %body, label %exit, !llvm.loop !7

; CHECK: exit:
exit:
  ret void
}

; Check that nested loops are visited.
;
; CHECK-LABEL: define void @nested(i32 %n) {
define void @nested(i32 %n) {
; CHECK: entry:
entry:
  br label %loop0.head

; CHECK: loop0.head:
loop0.head:
  %loop0.i = phi i32 [ 0, %entry ], [ %loop0.inc, %loop0.latch ]
  br label %loop1.head

; CHECK: loop1.head:
loop1.head:
  %loop1.i = phi i32 [ 0, %loop0.head ], [ %loop1.inc, %loop1.latch ]
  br label %loop2

; CHECK: loop2:
loop2:
  %loop2.i = phi i32 [ 0, %loop1.head ], [ %loop2.inc, %loop2 ]
  %loop2.inc = add nsw i32 %loop2.i, 1
  %loop2.cmp = icmp slt i32 %loop2.inc, %n
  ; CHECK: br i1 %loop2.cmp, label %loop2, label %loop1.latch, !prof !0, !llvm.loop ![[#NESTED_LOOP2:]]
  br i1 %loop2.cmp, label %loop2, label %loop1.latch, !prof !0

; CHECK: loop1.latch:
loop1.latch:
  %loop1.inc = add nsw i32 %loop1.i, 1
  %loop1.cmp = icmp slt i32 %loop1.inc, %n
  ; CHECK: br i1 %loop1.cmp, label %loop1.head, label %loop0.latch, !prof !0, !llvm.loop ![[#NESTED_LOOP1:]]
  br i1 %loop1.cmp, label %loop1.head, label %loop0.latch, !prof !0

; CHECK: loop0.latch:
loop0.latch:
  %loop0.inc = add nsw i32 %loop0.i, 1
  %loop0.cmp = icmp slt i32 %loop0.inc, %n
  ; CHECK: br i1 %loop0.cmp, label %loop0.head, label %exit, !prof !0, !llvm.loop ![[#NESTED_LOOP0:]]
  br i1 %loop0.cmp, label %loop0.head, label %exit, !prof !0

; CHECK: exit:
exit:
  ret void
}

; CHECK: !0 = !{!"branch_weights", i32 9, i32 1}
;
; CHECK: ![[#ESTIMABLE]] = distinct !{![[#ESTIMABLE]], ![[#ESTIMABLE_TC:]]}
; CHECK: ![[#ESTIMABLE_TC]] = !{!"llvm.loop.estimated_trip_count", i32 10}
;
; CHECK: ![[#NO_BRANCH_WEIGHTS]] = distinct !{![[#NO_BRANCH_WEIGHTS]], ![[#INESTIMABLE_TC:]]}
; CHECK: ![[#INESTIMABLE_TC]] = !{!"llvm.loop.estimated_trip_count"}
; CHECK: ![[#MULTI_LATCH]] = distinct !{![[#MULTI_LATCH]], ![[#INESTIMABLE_TC:]]}
;
; CHECK: ![[#VAL_ESTIMABLE]] = distinct !{![[#VAL_ESTIMABLE]], ![[#VAL_TC:]]}
; CHECK: ![[#VAL_TC]] = !{!"llvm.loop.estimated_trip_count", i32 5}
; CHECK: ![[#VAL_INESTIMABLE]] = distinct !{![[#VAL_INESTIMABLE]], ![[#VAL_TC:]]}
;
; CHECK: ![[#NO_VAL_ESTIMABLE]] = distinct !{![[#NO_VAL_ESTIMABLE]], ![[#ESTIMABLE_TC:]]}
; CHECK: ![[#NO_VAL_INESTIMABLE]] = distinct !{![[#NO_VAL_INESTIMABLE]], ![[#INESTIMABLE_TC:]]}
;
; CHECK: ![[#NESTED_LOOP2]] = distinct !{![[#NESTED_LOOP2]], ![[#ESTIMABLE_TC:]]}
; CHECK: ![[#NESTED_LOOP1]] = distinct !{![[#NESTED_LOOP1]], ![[#ESTIMABLE_TC:]]}
; CHECK: ![[#NESTED_LOOP0]] = distinct !{![[#NESTED_LOOP0]], ![[#ESTIMABLE_TC:]]}
!0 = !{!"branch_weights", i32 9, i32 1}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.estimated_trip_count", i32 5}
!3 = distinct !{!3, !2}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.estimated_trip_count"}
!7 = distinct !{!7, !6}
