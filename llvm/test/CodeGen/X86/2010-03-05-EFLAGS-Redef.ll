; RUN: llc < %s -verify-machineinstrs
;
; This test case is transformed into a single basic block by the machine
; branch folding pass. That makes a complete mess of the %eflags liveness, but
; we don't care about liveness this late anyway.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.2"

define i32 @main(i32 %argc, i8** nocapture %argv) ssp {
entry:
  br i1 undef, label %bb, label %bb2

bb:                                               ; preds = %entry
  br label %bb2

bb2:                                              ; preds = %bb, %entry
  br i1 undef, label %bb3, label %bb5

bb3:                                              ; preds = %bb2
  br label %bb5

bb5:                                              ; preds = %bb3, %bb2
  br i1 undef, label %bb.nph239, label %bb8

bb.nph239:                                        ; preds = %bb5
  unreachable

bb8:                                              ; preds = %bb5
  br i1 undef, label %bb.nph237, label %bb47

bb.nph237:                                        ; preds = %bb8
  unreachable

bb47:                                             ; preds = %bb8
  br i1 undef, label %bb49, label %bb48

bb48:                                             ; preds = %bb47
  unreachable

bb49:                                             ; preds = %bb47
  br i1 undef, label %bb51, label %bb50

bb50:                                             ; preds = %bb49
  ret i32 0

bb51:                                             ; preds = %bb49
  ret i32 0
}
