; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: Valid Region for Scop: bb14 => bb17

; Make sure we do not detect the larger region bb14->bb19 that contains
; a multi-dimensional memory access with a size of 'undef * undef'.

define void @hoge(ptr %arg) {
bb:
  br label %bb6

bb6:                                              ; preds = %bb
  %tmp = mul i64 undef, undef
  %tmp7 = add i64 %tmp, undef
  %tmp8 = add i64 %tmp7, 0
  %tmp9 = add i64 %tmp8, 8
  %tmp10 = sub i64 %tmp9, undef
  %tmp11 = getelementptr i8, ptr %arg, i64 %tmp10
  %tmp12 = getelementptr inbounds i8, ptr %tmp11, i64 4
  %tmp13 = getelementptr inbounds i8, ptr %tmp12, i64 20
  br label %bb14

bb14:                                             ; preds = %bb14, %bb6
  %tmp15 = phi i32 [ %tmp16, %bb14 ], [ 2, %bb6 ]
  %tmp16 = add nuw nsw i32 %tmp15, 1
  br i1 false, label %bb14, label %bb17

bb17:                                             ; preds = %bb14
  store i32 undef, ptr %tmp13, align 4
  br label %bb19

bb19:                                             ; preds = %bb17
  unreachable
}
