; RUN: opt -S -passes=loop-predication < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; XFAIL: *

declare void @llvm.experimental.deoptimize.isVoid(...)

; FIXME: Loop predication here inserts assume across the critical edge, and
; it leads to malformed IR (assume's condition does not dominate it).
define void @test_01(i1 %cond) {
; CHECK-LABEL: test
bb:
  %inst = call i1 @llvm.experimental.widenable.condition()
  br label %loop

unreached:                                              ; preds = %backedge
  unreachable

loop:                                              ; preds = %backedge, %bb
  %inst3 = phi i32 [ 0, %bb ], [ %inst4, %backedge ]
  %inst4 = add nsw i32 %inst3, 1
  br i1 %cond, label %backedge, label %guard_block

normal_ret:                                              ; preds = %loop
  ret void

backedge:                                              ; preds = %guard_block, %loop
  %inst7 = icmp sgt i32 %inst3, 137
  br i1 %inst7, label %unreached, label %loop

guard_block:                                              ; preds = %loop, %loop
  %inst9 = icmp ult i32 %inst4, 10000
  %inst10 = and i1 %inst9, %inst
  br i1 %inst10, label %backedge, label %deopt

deopt:                                             ; preds = %guard_block
  call void (...) @llvm.experimental.deoptimize.isVoid(i32 13) [ "deopt"() ]
  ret void

done:                                             ; preds = %loop
  ret void
}

declare i1 @llvm.experimental.widenable.condition()
