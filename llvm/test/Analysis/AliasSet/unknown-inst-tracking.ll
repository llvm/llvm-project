; RUN: opt -S -licm -simple-loop-unswitch < %s | FileCheck %s

; This test checks for a crash.  See PR32587.

@global = external global i32

declare i32 @f_1(i8, i32 returned)

define i32 @f_0() {
; CHECK-LABEL: @f_0(
bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %tmp = load i32, i32* @global
  %tmp2 = select i1 false, i16 1, i16 0
  br label %bb3

bb3:                                              ; preds = %bb3, %bb1
  %tmp4 = phi i8 [ 0, %bb1 ], [ %tmp6, %bb3 ]
  %tmp5 = icmp eq i16 %tmp2, 0
  %tmp6 = select i1 %tmp5, i8 %tmp4, i8 1
  %tmp7 = tail call i32 @f_1(i8 %tmp6, i32 1)
  br i1 false, label %bb1, label %bb3
}
