; RUN: opt -analyze -scalar-evolution -S < %s | FileCheck %s

; Every combination of
;  - starting at 0, 1, or %x
;  - steping by 1 or 2
;  - stopping at %n or %n*2
;  - using nsw, or not

; Some of these represent missed opportunities.

; CHECK: Determining loop execution counts for: @foo
; CHECK: Loop %loop: backedge-taken count is (-1 + %n)
; CHECK: Loop %loop: max backedge-taken count is 6
define void @foo(i4 %n) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 1
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @step2
; CHECK: Loop %loop: Unpredictable backedge-taken count.
; CHECK: Loop %loop: Unpredictable max backedge-taken count.
define void @step2(i4 %n) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 2
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @start1
; CHECK: Loop %loop: backedge-taken count is (-2 + (2 smax %n))
; CHECK: Loop %loop: max backedge-taken count is 5
define void @start1(i4 %n) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 1, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 1
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @start1_step2
; CHECK: Loop %loop: Unpredictable backedge-taken count.
; CHECK: Loop %loop: Unpredictable max backedge-taken count.
define void @start1_step2(i4 %n) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 1, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 2
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @startx
; CHECK: Loop %loop: backedge-taken count is (-1 + (-1 * %x) + ((1 + %x) smax %n))
; CHECK: Loop %loop: max backedge-taken count is -1
define void @startx(i4 %n, i4 %x) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ %x, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 1
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @startx_step2
; CHECK: Loop %loop: Unpredictable backedge-taken count.
; CHECK: Loop %loop: Unpredictable max backedge-taken count.
define void @startx_step2(i4 %n, i4 %x) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ %x, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 2
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @nsw
; CHECK: Loop %loop: backedge-taken count is (-1 + %n)
; CHECK: Loop %loop: max backedge-taken count is 6
define void @nsw(i4 %n) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 1
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; If %n is INT4_MAX, %i.next will wrap. The nsw bit says that the
; result is undefined. Therefore, after the loop's second iteration,
; we are free to assume that the loop exits. This is valid because:
; (a) %i.next is a poison value after the second iteration, which can
; also be considered an undef value.
; (b) the return instruction enacts a side effect that is control
; dependent on the poison value.
;
; CHECK-LABEL: nsw_step2
; CHECK: Determining loop execution counts for: @nsw_step2
; CHECK: Loop %loop: backedge-taken count is ((-1 + %n) /u 2)
; CHECK: Loop %loop: max backedge-taken count is 2
define void @nsw_step2(i4 %n) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 2
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK-LABEL: nsw_start1
; CHECK: Determining loop execution counts for: @nsw_start1
; CHECK: Loop %loop: backedge-taken count is (-2 + (2 smax %n))
; CHECK: Loop %loop: max backedge-taken count is 5
define void @nsw_start1(i4 %n) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 1, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 1
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @nsw_start1_step2
; CHECK: Loop %loop: backedge-taken count is ((-2 + (3 smax %n)) /u 2)
; CHECK: Loop %loop: max backedge-taken count is 2
define void @nsw_start1_step2(i4 %n) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 1, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 2
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @nsw_startx
; CHECK: Loop %loop: backedge-taken count is (-1 + (-1 * %x) + ((1 + %x) smax %n))
; CHECK: Loop %loop: max backedge-taken count is -1
define void @nsw_startx(i4 %n, i4 %x) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ %x, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 1
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @nsw_startx_step2
; CHECK: Loop %loop: backedge-taken count is ((-1 + (-1 * %x) + ((2 + %x) smax %n)) /u 2)
; CHECK: Loop %loop: max backedge-taken count is 7
define void @nsw_startx_step2(i4 %n, i4 %x) {
entry:
  %s = icmp sgt i4 %n, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ %x, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 2
  %t = icmp slt i4 %i.next, %n
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even
; CHECK: Loop %loop: backedge-taken count is (-1 + (2 * %n))
; CHECK: Loop %loop: max backedge-taken count is 5
define void @even(i4 %n) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 1
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_step2
; CHECK: Loop %loop: backedge-taken count is ((-1 + (2 * %n)) /u 2)
; CHECK: Loop %loop: max backedge-taken count is 2
define void @even_step2(i4 %n) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 2
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_start1
; CHECK: Loop %loop: backedge-taken count is (-2 + (2 smax (2 * %n)))
; CHECK: Loop %loop: max backedge-taken count is 4
define void @even_start1(i4 %n) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 1, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 1
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_start1_step2
; CHECK: Loop %loop: backedge-taken count is ((-2 + (3 smax (2 * %n))) /u 2)
; CHECK: Loop %loop: max backedge-taken count is 2
define void @even_start1_step2(i4 %n) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 1, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 2
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_startx
; CHECK: Loop %loop: backedge-taken count is (-1 + (-1 * %x) + ((1 + %x) smax (2 * %n)))
; CHECK: Loop %loop: max backedge-taken count is -2
define void @even_startx(i4 %n, i4 %x) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ %x, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 1
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_startx_step2
; CHECK: Loop %loop: backedge-taken count is ((-1 + (-1 * %x) + ((2 + %x) smax (2 * %n))) /u 2)
; CHECK: Loop %loop: max backedge-taken count is 7
define void @even_startx_step2(i4 %n, i4 %x) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ %x, %entry ], [ %i.next, %loop ]
  %i.next = add i4 %i, 2
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_nsw
; CHECK: Loop %loop: backedge-taken count is (-1 + (2 * %n))
; CHECK: Loop %loop: max backedge-taken count is 5
define void @even_nsw(i4 %n) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 1
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_nsw_step2
; CHECK: Loop %loop: backedge-taken count is ((-1 + (2 * %n)) /u 2)
; CHECK: Loop %loop: max backedge-taken count is 2
define void @even_nsw_step2(i4 %n) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 0, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 2
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_nsw_start1
; CHECK: Loop %loop: backedge-taken count is (-2 + (2 smax (2 * %n)))
; CHECK: Loop %loop: max backedge-taken count is 4
define void @even_nsw_start1(i4 %n) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 1, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 1
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_nsw_start1_step2
; CHECK: Loop %loop: backedge-taken count is ((-2 + (3 smax (2 * %n))) /u 2)
; CHECK: Loop %loop: max backedge-taken count is 2
define void @even_nsw_start1_step2(i4 %n) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ 1, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 2
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_nsw_startx
; CHECK: Loop %loop: backedge-taken count is (-1 + (-1 * %x) + ((1 + %x) smax (2 * %n)))
; CHECK: Loop %loop: max backedge-taken count is -2
define void @even_nsw_startx(i4 %n, i4 %x) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ %x, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 1
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}

; CHECK: Determining loop execution counts for: @even_nsw_startx_step2
; CHECK: Loop %loop: backedge-taken count is ((-1 + (-1 * %x) + ((2 + %x) smax (2 * %n))) /u 2)
; CHECK: Loop %loop: max backedge-taken count is 7
define void @even_nsw_startx_step2(i4 %n, i4 %x) {
entry:
  %m = shl i4 %n, 1
  %s = icmp sgt i4 %m, 0
  br i1 %s, label %loop, label %exit
loop:
  %i = phi i4 [ %x, %entry ], [ %i.next, %loop ]
  %i.next = add nsw i4 %i, 2
  %t = icmp slt i4 %i.next, %m
  br i1 %t, label %loop, label %exit
exit:
  ret void
}
