; RUN: opt -verify-loop-info -irce-print-changed-loops -passes=irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes='require<branch-prob>,irce' -S < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

; IRCE should fail here because the preheader's exiting value is a phi from the
; loop, and this value cannot be expanded at loop's preheader.

; CHECK-NOT:   irce: in function test_01: constrained Loop
; CHECK-NOT:   irce: in function test_02: constrained Loop
; CHECK-LABEL: irce: in function test_03: constrained Loop

define void @test_01() {

; CHECK-NOT:   irce: in function test_01: constrained Loop

; CHECK-LABEL: test_01
; CHECK-NOT:   preloop
; CHECK-NOT:   postloop
; CHECK-NOT:   br i1 false
; CHECK-NOT:   br i1 true

entry:
  br label %loop

exit:                                       ; preds = %guarded, %loop
  ret void

loop:                                      ; preds = %guarded, %entry
  %iv = phi i64 [ 380, %entry ], [ %limit, %guarded ]
  %bad_phi = phi i32 [ 3, %entry ], [ %bad_phi.next, %guarded ]
  %bad_phi.next = add nuw nsw i32 %bad_phi, 1
  %iv.next = add nuw nsw i64 %iv, 1
  %rc = icmp slt i64 %iv.next, 5
  br i1 %rc, label %guarded, label %exit

guarded:
  %limit = add nsw i64 %iv, -1
  %tmp5 = add nuw nsw i32 %bad_phi, 8
  %tmp6 = zext i32 %tmp5 to i64
  %tmp7 = icmp eq i64 %limit, %tmp6
  br i1 %tmp7, label %exit, label %loop
}

; This test should fail because we are unable to prove that the division is
; safe to expand it to preheader: if we exit by maybe_exit condition, it is
; unsafe to execute it there.

define void @test_02(ptr %p1, ptr %p2, i1 %maybe_exit) {

; CHECK-LABEL: test_02
; CHECK-NOT:   preloop
; CHECK-NOT:   postloop
; CHECK-NOT:   br i1 false
; CHECK-NOT:   br i1 true


entry:
  %num = load i64, ptr %p1, align 4, !range !0
  %denom = load i64, ptr %p2, align 4, !range !0
  br label %loop

exit:                                       ; preds = %guarded, %loop
  ret void

loop:                                      ; preds = %guarded, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %guarded ]
  %iv.next = add nuw nsw i64 %iv, 1
  br i1 %maybe_exit, label %range_check, label %exit

range_check:
  %div_result = udiv i64 %num, %denom
  %rc = icmp slt i64 %iv.next, %div_result
  br i1 %rc, label %guarded, label %exit

guarded:
  %gep = getelementptr i64, ptr %p1, i64 %iv.next
  %loaded = load i64, ptr %gep, align 4
  %tmp7 = icmp slt i64 %iv.next, 1000
  br i1 %tmp7, label %loop, label %exit
}

define void @test_03(ptr %p1, ptr %p2, i1 %maybe_exit) {

; Show that IRCE would hit test_02 if the division was safe (denom not zero).

; CHECK-LABEL: test_03
; CHECK:       entry:
; CHECK-NEXT:    %num = load i64, ptr %p1, align 4
; CHECK-NEXT:    [[DIV:%[^ ]+]] = udiv i64 %num, 13
; CHECK-NEXT:    [[DIV_MINUS_1:%[^ ]+]] = add nsw i64 [[DIV]], -1
; CHECK-NEXT:    %exit.mainloop.at = call i64 @llvm.smax.i64(i64 [[DIV_MINUS_1]], i64 0)
; CHECK-NEXT:    [[COMP2:%[^ ]+]] = icmp slt i64 0, %exit.mainloop.at
; CHECK-NEXT:    br i1 [[COMP2]], label %loop.preheader, label %main.pseudo.exit
; CHECK-NOT:     preloop
; CHECK:       loop:
; CHECK-NEXT:    %iv = phi i64 [ %iv.next, %guarded ], [ 0, %loop.preheader ]
; CHECK-NEXT:    %iv.next = add nuw nsw i64 %iv, 1
; CHECK-NEXT:    %rc = icmp slt i64 %iv.next, %div_result
; CHECK-NEXT:    %or.cond = select i1 %maybe_exit, i1 true, i1 false
; CHECK-NEXT:    br i1 %or.cond, label %guarded, label %exit.loopexit1
; CHECK:       guarded:
; CHECK-NEXT:    %gep = getelementptr i64, ptr %p1, i64 %iv.next
; CHECK-NEXT:    %loaded = load i64, ptr %gep, align 4
; CHECK-NEXT:    %tmp7 = icmp slt i64 %iv.next, 1000
; CHECK-NEXT:    [[EXIT_MAIN_LOOP:%[^ ]+]] = icmp slt i64 %iv.next, %exit.mainloop.at
; CHECK-NEXT:    br i1 [[EXIT_MAIN_LOOP]], label %loop, label %main.exit.selector
; CHECK:       postloop

entry:
  %num = load i64, ptr %p1, align 4, !range !0
  br label %loop

exit:                                       ; preds = %guarded, %loop
  ret void

loop:                                      ; preds = %guarded, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %guarded ]
  %iv.next = add nuw nsw i64 %iv, 1
  br i1 %maybe_exit, label %range_check, label %exit

range_check:
  %div_result = udiv i64 %num, 13
  %rc = icmp slt i64 %iv.next, %div_result
  br i1 %rc, label %guarded, label %exit

guarded:
  %gep = getelementptr i64, ptr %p1, i64 %iv.next
  %loaded = load i64, ptr %gep, align 4
  %tmp7 = icmp slt i64 %iv.next, 1000
  br i1 %tmp7, label %loop, label %exit
}

!0 = !{i64 0, i64 100}
