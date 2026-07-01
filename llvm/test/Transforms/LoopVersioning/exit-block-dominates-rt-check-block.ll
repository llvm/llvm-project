; This test ensures loop versioning does not produce an invalid dominator tree
; if the exit block of the loop (bb0) dominates the runtime check block
; (bb1 will become the runtime check block).

; RUN: opt -passes=loop-distribute -enable-loop-distribute -verify-dom-info -S -o - %s > %t
; RUN: opt -passes=loop-simplify,loop-distribute -enable-loop-distribute -verify-dom-info -S -o - %s > %t
; RUN: FileCheck --check-prefix CHECK-VERSIONING -input-file %t %s

; RUN: opt -passes=loop-versioning -verify-dom-info -S -o - %s > %t
; RUN: opt -passes=loop-simplify,loop-versioning -verify-dom-info -S -o - %s > %t
; RUN: FileCheck --check-prefix CHECK-VERSIONING -input-file %t %s

@c1 = external global i16

define void @f(i16 %a, ptr %p) {
  br label %bb0

bb0:
  br label %bb1

bb1:
  %tmp1 = load i16, ptr @c1
  br label %bb2

bb2:
  %tmp2 = phi i16 [ %tmp1, %bb1 ], [ %tmp3, %bb2 ]
  %iv = phi i32 [ 0, %bb1 ], [ %iv.next, %bb2 ]
  %tmp4 = getelementptr inbounds i32, ptr %p, i32 %iv
  store i32 1, ptr %tmp4
  %tmp5 = getelementptr inbounds i32, ptr %p, i32 4
  store i32 0, ptr %tmp5
  %tmp3 = add i16 %tmp2, 1
  store i16 %tmp2, ptr @c1
  %iv.next = add i32 %iv, 1
  %tmp6 = icmp sle i16 %tmp3, 0
  br i1 %tmp6, label %bb2, label %bb0
}

; Simple check to make sure loop versioning happened.
; CHECK-VERSIONING: bb2.lver.check:
