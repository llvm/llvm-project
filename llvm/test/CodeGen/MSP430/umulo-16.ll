; RUN: llc < %s -mtriple=msp430 | FileCheck %s
target datalayout = "e-m:e-p:16:16-i32:16:32-a:16-n8:16"
target triple = "msp430"

define void @foo(i16 %arg) unnamed_addr {
entry-block:
  br i1 undef, label %bb2, label %bb3

bb2:                                              ; preds = %entry-block
  unreachable

bb3:                                              ; preds = %entry-block
  %0 = call { i16, i1 } @llvm.umul.with.overflow.i16(i16 undef, i16 %arg)
; CHECK: call
  %1 = extractvalue { i16, i1 } %0, 1
  %2 = call i1 @llvm.expect.i1(i1 %1, i1 false)
  br i1 %2, label %panic, label %bb5

bb5:                                              ; preds = %bb3
  unreachable

panic:                                            ; preds = %bb3
  unreachable
}

; Function Attrs: nounwind readnone
declare i1 @llvm.expect.i1(i1, i1) #0

; Function Attrs: nounwind readnone
declare { i16, i1 } @llvm.umul.with.overflow.i16(i16, i16) #0

attributes #0 = { nounwind readnone }
