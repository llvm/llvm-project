; RUN: opt %s -passes='print<scalar-evolution>' -scalar-evolution-classify-expressions=0 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Tests for multi-exit loops.
;; Computing exit counts using AddRec nuw/nsw flags

;; ============================================================
;; Test 1: multi-exit, step +8, nuw, ult with symbolic bound
;; With nuw fallback: exit count for the ult exit is computable
;; ============================================================
; CHECK: Determining loop execution counts for: @multi_exit_step8_nuw_ult
; CHECK: Loop %loop: <multiple exits> backedge-taken count is (((7 + (-1 * %start) + (%start umax %bound)) /u 8) umin_seq (-1 + (1 umax %n)))
; CHECK:   exit count for loop: ((7 + (-1 * %start) + (%start umax %bound)) /u 8)
; CHECK:   exit count for latch: (-1 + (1 umax %n))

define void @multi_exit_step8_nuw_ult(i32 %start, i32 %bound, i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %i = phi i32 [0, %entry], [%i.next, %latch]
  %cmp = icmp ult i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add nuw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

;; ============================================================
;; Test 2: multi-exit, step +8, NO nuw, ult — stays unpredictable
;; Negative test: without nuw, canIVOverflowOnLT returns true
;; and there's no fallback.
;; ============================================================
; CHECK: Determining loop execution counts for: @multi_exit_step8_no_nuw_ult
; CHECK: Loop %loop: <multiple exits> Unpredictable backedge-taken count.
; CHECK:   exit count for loop: ***COULDNOTCOMPUTE***
; CHECK:   exit count for latch: (-1 + (1 umax %n))

define void @multi_exit_step8_no_nuw_ult(i32 %start, i32 %bound, i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %i = phi i32 [0, %entry], [%i.next, %latch]
  %cmp = icmp ult i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

;; ============================================================
;; Test 3: multi-exit, step +8, nuw+nsw, slt with symbolic bound
;; nsw fallback makes signed exit count computable
;; ============================================================
; CHECK: Determining loop execution counts for: @multi_exit_step8_nsw_slt
; CHECK: Loop %loop: <multiple exits> backedge-taken count is (((7 + (-1 * %start) + (%start smax %bound)) /u 8) umin_seq (-1 + (1 umax %n)))
; CHECK:   exit count for loop: ((7 + (-1 * %start) + (%start smax %bound)) /u 8)
; CHECK:   exit count for latch: (-1 + (1 umax %n))

define void @multi_exit_step8_nsw_slt(i32 %start, i32 %bound, i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %i = phi i32 [0, %entry], [%i.next, %latch]
  %cmp = icmp slt i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add nuw nsw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

;; ============================================================
;; Test 4: multi-exit, step +8, nuw only, slt — stays unpredictable
;; Negative test: nuw does NOT help signed predicates (needs nsw).
;; INT_MAX + 8 with nuw is valid unsigned but wraps signed.
;; ============================================================
; CHECK: Determining loop execution counts for: @multi_exit_step8_nuw_only_slt
; CHECK:   exit count for loop: ***COULDNOTCOMPUTE***

define void @multi_exit_step8_nuw_only_slt(i32 %start, i32 %bound, i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %i = phi i32 [0, %entry], [%i.next, %latch]
  %cmp = icmp slt i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add nuw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

;; ============================================================
;; Test 5: multi-exit, step -4, nsw, sgt with symbolic bound
;; howManyGreaterThans nsw fallback
;; ============================================================
; CHECK: Determining loop execution counts for: @multi_exit_neg_step4_nsw_sgt
; CHECK: Loop %loop: <multiple exits> backedge-taken count is (((3 + (-1 * (%start smin %bound)) + %start) /u 4) umin_seq (-1 + (1 umax %n)))
; CHECK:   exit count for loop: ((3 + (-1 * (%start smin %bound)) + %start) /u 4)
; CHECK:   exit count for latch: (-1 + (1 umax %n))

define void @multi_exit_neg_step4_nsw_sgt(i32 %start, i32 %bound, i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %i = phi i32 [0, %entry], [%i.next, %latch]
  %cmp = icmp sgt i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add nsw i32 %iv, -4
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

;; ============================================================
;; Test 6: multi-exit, step -4, NO nsw, sgt — stays unpredictable
;; Negative test: without nsw, no fallback for signed GT.
;; ============================================================
; CHECK: Determining loop execution counts for: @multi_exit_neg_step4_no_nsw_sgt
; CHECK:   exit count for loop: ***COULDNOTCOMPUTE***

define void @multi_exit_neg_step4_no_nsw_sgt(i32 %start, i32 %bound, i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %i = phi i32 [0, %entry], [%i.next, %latch]
  %cmp = icmp sgt i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add i32 %iv, -4
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %exit

exit:
  ret void
}

;; ============================================================
;; Test 7: single-exit, step +8, nuw, ult — already works (baseline)
;; ControlsOnlyExit=true, so NoWrap=true directly.
;; Included to verify the patch doesn't regress this path.
;; ============================================================
; CHECK: Determining loop execution counts for: @single_exit_step8_nuw_ult
; CHECK: Loop %loop: backedge-taken count is ((7 + (-1 * %start) + (%start umax %bound)) /u 8)

define void @single_exit_step8_nuw_ult(i32 %start, i32 %bound) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %cmp = icmp ult i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add nuw i32 %iv, 8
  br label %loop

exit:
  ret void
}

;; ============================================================
;; Test 8: single-exit, step +8, NO nuw, ult — unpredictable
;; Even with ControlsOnlyExit=true, NoWrap=false without flags.
;; ============================================================
; CHECK: Determining loop execution counts for: @single_exit_step8_no_nuw_ult
; CHECK: Loop %loop: Unpredictable backedge-taken count.

define void @single_exit_step8_no_nuw_ult(i32 %start, i32 %bound) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %cmp = icmp ult i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add i32 %iv, 8
  br label %loop

exit:
  ret void
}

;; ============================================================
;; Test 9: single-exit, step +8, nuw+nsw, slt — computable
;; ControlsOnlyExit=true + nsw → NoWrap=true for signed.
;; ============================================================
; CHECK: Determining loop execution counts for: @single_exit_step8_nsw_slt
; CHECK: Loop %loop: backedge-taken count is ((7 + (-1 * %start) + (%start smax %bound)) /u 8)

define void @single_exit_step8_nsw_slt(i32 %start, i32 %bound) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %cmp = icmp slt i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add nuw nsw i32 %iv, 8
  br label %loop

exit:
  ret void
}

;; ============================================================
;; Test 10: single-exit, step -4, nsw, sgt — computable
;; ControlsOnlyExit=true + nsw → NoWrap=true for signed GT.
;; ============================================================
; CHECK: Determining loop execution counts for: @single_exit_neg_step4_nsw_sgt
; CHECK: Loop %loop: backedge-taken count is ((3 + (-1 * (%start smin %bound)) + %start) /u 4)

define void @single_exit_neg_step4_nsw_sgt(i32 %start, i32 %bound) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %cmp = icmp sgt i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add nsw i32 %iv, -4
  br label %loop

exit:
  ret void
}

;; ============================================================
;; Test 11: single-exit, step +8, NO nsw, slt — unpredictable
;; No flags → NoWrap=false, canIVOverflowOnLT=true for symbolic.
;; ============================================================
; CHECK: Determining loop execution counts for: @single_exit_step8_no_nsw_slt
; CHECK: Loop %loop: Unpredictable backedge-taken count.

define void @single_exit_step8_no_nsw_slt(i32 %start, i32 %bound) {
entry:
  br label %loop

loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %latch]
  %cmp = icmp slt i32 %iv, %bound
  br i1 %cmp, label %latch, label %exit

latch:
  %iv.next = add i32 %iv, 8
  br label %loop

exit:
  ret void
}
