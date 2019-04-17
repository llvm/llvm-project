; RUN: opt -S -indvars < %s | FileCheck %s

define i16 @fn1() {
; CHECK-LABEL: @fn1(
entry:
  br label %bb1

bb1:
  %i = phi i16 [ 0, %entry ], [ 1, %bb1 ]
  %storemerge = phi i16 [ %storemerge2, %bb1 ], [ 0, %entry ]
  %storemerge2 = phi i16 [ 10, %entry ], [ 200, %bb1 ]
  %tmp10 = icmp eq i16 %i, 1
  br i1 %tmp10, label %bb5, label %bb1

bb5:
  %storemerge.lcssa = phi i16 [ %storemerge, %bb1 ]
; CHECK: ret i16 10
  ret i16 %storemerge.lcssa
}

define i16 @fn2() {
; CHECK-LABEL: @fn2(
entry:
  br label %bb1

bb1:
  %canary = phi i16 [ 0, %entry ], [ %canary.inc, %bb1 ]
  %i = phi i16 [ 0, %entry ], [ %storemerge, %bb1 ]
  %storemerge = phi i16 [ 0, %bb1 ], [ 10, %entry ]
  %canary.inc = add i16 %canary, 1
  %_tmp10 = icmp eq i16 %i, 10
  br i1 %_tmp10, label %bb5, label %bb1

bb5:
; CHECK: ret i16 1
  ret i16 %canary
}
