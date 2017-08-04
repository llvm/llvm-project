; RUN: opt %loadPolly -analyze -polly-detect \
; RUN:     -pass-remarks-missed="polly-detect" \
; RUN:     < %s 2>&1| FileCheck %s

; CHECK: remark: <unknown>:0:0: Loop cannot be handled because not all latches are part of loop region.
; CHECK: remark: <unknown>:0:0: Loop cannot be handled because not all latches are part of loop region.


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @baz(i32 %before) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb
  br label %bb2

bb2:                                              ; preds = %bb8, %bb7, %bb2, %bb1
  %tmp = phi i32 [ %before, %bb1 ], [ 0, %bb8 ], [ %tmp4, %bb7 ], [ %tmp4, %bb2 ]
  %tmp3 = or i32 undef, undef
  %tmp4 = udiv i32 %tmp3, 10
  %tmp5 = trunc i32 undef to i8
  %tmp6 = icmp eq i8 %tmp5, 0
  br i1 %tmp6, label %bb7, label %bb2

bb7:                                              ; preds = %bb2
  br i1 undef, label %bb8, label %bb2

bb8:                                              ; preds = %bb7
  br i1 undef, label %bb9, label %bb2

bb9:                                              ; preds = %bb8
  unreachable
}
