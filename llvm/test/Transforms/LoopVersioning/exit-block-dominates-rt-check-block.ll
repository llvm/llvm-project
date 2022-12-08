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

define void @f(i16 %a) {
  br label %bb0

bb0:
  br label %bb1

bb1:
  %tmp1 = load i16, i16* @c1
  br label %bb2

bb2:
  %tmp2 = phi i16 [ %tmp1, %bb1 ], [ %tmp3, %bb2 ]
  %tmp4 = getelementptr inbounds [1 x i32], [1 x i32]* undef, i32 0, i32 4
  store i32 1, i32* %tmp4
  %tmp5 = getelementptr inbounds [1 x i32], [1 x i32]* undef, i32 0, i32 9
  store i32 0, i32* %tmp5
  %tmp3 = add i16 %tmp2, 1
  store i16 %tmp2, i16* @c1
  %tmp6 = icmp sle i16 %tmp3, 0
  br i1 %tmp6, label %bb2, label %bb0
}

; Simple check to make sure loop versioning happened.
; CHECK-VERSIONING: bb2.lver.check:
