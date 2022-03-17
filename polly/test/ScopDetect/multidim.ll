; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: Valid Region for Scop: bb19 => bb20

; Make sure we do not crash in this test case.

define void @hoge(i8* %arg)  {
bb:
  br label %bb9

bb9:                                              ; preds = %bb
  %tmp = add i64 undef, 4
  %tmp10 = zext i32 undef to i64
  %tmp11 = mul i64 %tmp, %tmp10
  %tmp12 = add i64 %tmp11, undef
  %tmp13 = add i64 %tmp12, undef
  %tmp14 = add i64 %tmp13, 8
  %tmp15 = sub i64 %tmp14, undef
  %tmp16 = getelementptr i8, i8* %arg, i64 %tmp15
  %tmp17 = getelementptr inbounds i8, i8* %tmp16, i64 4
  %tmp18 = getelementptr inbounds i8, i8* %tmp17, i64 20
  br label %bb19

bb19:                                             ; preds = %bb19, %bb9
  br i1 false, label %bb19, label %bb20

bb20:                                             ; preds = %bb19
  %tmp21 = getelementptr inbounds i8, i8* %tmp18, i64 4
  %tmp22 = bitcast i8* %tmp21 to i32*
  %tmp23 = load i32, i32* %tmp22
  br label %bb24

bb24:                                             ; preds = %bb20
  unreachable
}
