; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
;
; Verify that we do not model atomic memory accesses. We did not reason about
; how to handle them correctly and the Alias Set Tracker models some of them
; only as Unknown Instructions, which we do not know how to handle either.;
;
; CHECK-NOT: Valid
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@global = external global i64, align 8

declare void @foo55()

define void @blam107() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  %tmp = load atomic i8, ptr @global acquire, align 8
  br i1 false, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  tail call void @foo55() #6
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  unreachable
}

