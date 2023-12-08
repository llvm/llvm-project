; RUN: llc --start-before loop-reduce --stop-after loop-reduce %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @hoge
define void @hoge() {
bb:
  %tmp = sext i32 undef to i64
  %tmp3 = sub nsw i64 0, %tmp
  br label %bb4

bb4:                                              ; preds = %bb20, %bb
  %tmp5 = getelementptr inbounds double, ptr undef, i64 undef
  %tmp6 = getelementptr inbounds double, ptr %tmp5, i64 %tmp3
  br label %bb7

bb7:                                              ; preds = %bb7, %bb4
  %tmp8 = phi ptr [ %tmp10, %bb7 ], [ %tmp6, %bb4 ]
  %tmp9 = load double, ptr %tmp8
  %tmp10 = getelementptr inbounds double, ptr %tmp8, i64 1
  br i1 true, label %bb11, label %bb7

bb11:                                             ; preds = %bb7
  br i1 undef, label %bb20, label %bb12

bb12:                                             ; preds = %bb11
  br label %bb13

bb13:                                             ; preds = %bb13, %bb12
  %tmp14 = phi ptr [ %tmp18, %bb13 ], [ %tmp10, %bb12 ]
  %tmp15 = load double, ptr %tmp14, align 8
  %tmp16 = getelementptr inbounds double, ptr %tmp14, i64 1
  %tmp17 = load double, ptr %tmp16, align 8
  %tmp18 = getelementptr inbounds double, ptr %tmp14, i64 8
  br i1 true, label %bb19, label %bb13

bb19:                                             ; preds = %bb13
  br label %bb20

bb20:                                             ; preds = %bb19, %bb11
  br label %bb4
}
