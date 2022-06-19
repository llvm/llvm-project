; RUN: opt %loadPolly -polly-codegen \
; RUN:     -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.wombat = type {[4 x i32]}

; CHECK-NOT:  polly.preload.begin:
; CHECK-NOT:    %polly.access.B
; CHECK-NOT:    %polly.access.B.load

; CHECK: polly.split_new_and_old

; CHECK: polly.stmt.bb3.entry:                             ; preds = %polly.start
; CHECK:   br label %polly.stmt.bb3

; CHECK: polly.stmt.bb3:                                   ; preds = %polly.stmt.bb3.entry
; CHECK:   br i1 true, label %polly.stmt.bb4, label %polly.stmt.bb5

; CHECK: polly.stmt.bb4:                                   ; preds = %polly.stmt.bb3
; CHECK:   br label %polly.stmt.bb13.exit

; CHECK: polly.stmt.bb5:                                   ; preds = %polly.stmt.bb3
; CHECK:   load i32, i32* %B

; Function Attrs: nounwind uwtable
define void @quux(%struct.wombat* %arg, i32* %B) {
bb:
  br i1 undef, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %tmp = phi i1 [ true, %bb ], [ undef, %bb1 ]
  br label %bb3

bb3:                                              ; preds = %bb13, %bb2
  br i1 %tmp, label %bb4, label %bb5

bb4:                                              ; preds = %bb3
  br label %bb13

bb5:                                              ; preds = %bb3
  %tmp7 = load i32, i32* %B
  %tmp12 = getelementptr inbounds %struct.wombat, %struct.wombat* %arg, i64 0, i32 0, i64 0
  store i32 %tmp7, i32* %tmp12
  br label %bb13

bb13:                                             ; preds = %bb5, %bb4
  br i1 false, label %bb3, label %bb14

bb14:                                             ; preds = %bb13
  br label %bb15

bb15:                                             ; preds = %bb14
  ret void
}
