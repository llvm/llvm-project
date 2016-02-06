; RUN: opt -S -indvars < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @main(i16 %in) {
; CHECK-LABEL: @main(
  br label %bb2

bb2:                                              ; preds = %bb1.i, %bb2, %0
  %_tmp44.i = icmp slt i16 %in, 2
  br i1 %_tmp44.i, label %bb1.i, label %bb2

bb1.i:                                            ; preds = %bb1.i, %bb2
  %_tmp25.i = phi i16 [ %in, %bb2 ], [ %_tmp6.i, %bb1.i ]
  %_tmp6.i = add nsw i16 %_tmp25.i, 1
  %_tmp10.i = icmp sge i16 %_tmp6.i, 2
  %exitcond.i = icmp eq i16 %_tmp6.i, 2
  %or.cond = and i1 %_tmp10.i, %exitcond.i
  br i1 %or.cond, label %bb2, label %bb1.i
}
