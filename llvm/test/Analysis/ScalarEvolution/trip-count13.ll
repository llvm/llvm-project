; RUN: opt -S -analyze -scalar-evolution < %s | FileCheck %s

define void @u_0(i8 %rhs) {
; E.g.: %rhs = 255, %start = 99, backedge taken 156 times
entry:
  %start = add i8 %rhs, 100
  br label %loop

loop:
  %iv = phi i8 [ %start, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add nuw i8 %iv, 1  ;; Note: this never unsigned-wraps
  %iv.cmp = icmp ult i8 %iv, %rhs
  br i1 %iv.cmp, label %loop, label %leave

; CHECK-LABEL: Determining loop execution counts for: @u_0
; CHECK-NEXT: Loop %loop: backedge-taken count is (-100 + (-1 * %rhs) + ((100 + %rhs) umax %rhs))
; CHECK-NEXT: Loop %loop: max backedge-taken count is -100, actual taken count either this or zero.

leave:
  ret void
}

define void @u_1(i8 %start) {
entry:
; E.g.: %start = 99, %rhs = 255, backedge taken 156 times
  %rhs = add i8 %start, -100
  br label %loop

loop:
  %iv = phi i8 [ %start, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add nuw i8 %iv, 1  ;; Note: this never unsigned-wraps
  %iv.cmp = icmp ult i8 %iv, %rhs
  br i1 %iv.cmp, label %loop, label %leave

; CHECK-LABEL: Determining loop execution counts for: @u_1
; CHECK-NEXT: Loop %loop: backedge-taken count is ((-1 * %start) + ((-100 + %start) umax %start))
; CHECK-NEXT: Loop %loop: max backedge-taken count is -100, actual taken count either this or zero.

leave:
  ret void
}

define void @s_0(i8 %rhs) {
entry:
; E.g.: %rhs = 127, %start = -29, backedge taken 156 times
  %start = add i8 %rhs, 100
  br label %loop

loop:
  %iv = phi i8 [ %start, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add nsw i8 %iv, 1  ;; Note: this never signed-wraps
  %iv.cmp = icmp slt i8 %iv, %rhs
  br i1 %iv.cmp, label %loop, label %leave

; CHECK-LABEL: Determining loop execution counts for: @s_0
; CHECK-NEXT: Loop %loop: backedge-taken count is (-100 + (-1 * %rhs) + ((100 + %rhs) smax %rhs))
; CHECK-NEXT: Loop %loop: max backedge-taken count is -100, actual taken count either this or zero.

leave:
  ret void
}

define void @s_1(i8 %start) {
entry:
; E.g.: start = -29, %rhs = 127, %backedge taken 156 times
  %rhs = add i8 %start, -100
  br label %loop

loop:
  %iv = phi i8 [ %start, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add nsw i8 %iv, 1
  %iv.cmp = icmp slt i8 %iv, %rhs
  br i1 %iv.cmp, label %loop, label %leave

; CHECK-LABEL: Determining loop execution counts for: @s_1
; CHECK-NEXT: Loop %loop: backedge-taken count is ((-1 * %start) + ((-100 + %start) smax %start))
; CHECK-NEXT: Loop %loop: max backedge-taken count is -100, actual taken count either this or zero.

leave:
  ret void
}
