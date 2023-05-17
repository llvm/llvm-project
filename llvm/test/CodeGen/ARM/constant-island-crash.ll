; RUN: llc < %s
; No FileCheck - testing for crash.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv5e-none-linux-gnueabi"

%struct.blam = type { [4 x %struct.eggs], [6 x [15 x i16]], [6 x i32], i32, i32, i32, i32, i32, i32, %struct.eggs, [4 x %struct.eggs], [4 x %struct.eggs], [4 x i32], i32, i32, i32, [4 x %struct.eggs], [4 x %struct.eggs], i32, %struct.eggs, i32 }
%struct.eggs = type { i32, i32 }

define void @spam(ptr %arg, i32 %arg1) {
bb:
  %tmp = getelementptr inbounds %struct.blam, ptr %arg, i32 undef, i32 2, i32 %arg1
  switch i32 %arg1, label %bb8 [
    i32 0, label %bb2
    i32 1, label %bb3
    i32 2, label %bb4
    i32 3, label %bb5
    i32 4, label %bb6
    i32 5, label %bb7
  ]

bb2:                                              ; preds = %bb
  unreachable

bb3:                                              ; preds = %bb
  unreachable

bb4:                                              ; preds = %bb
  unreachable

bb5:                                              ; preds = %bb
  unreachable

bb6:                                              ; preds = %bb
  unreachable

bb7:                                              ; preds = %bb
  unreachable

bb8:                                              ; preds = %bb
  store i32 1, ptr %tmp, align 4
  unreachable
}
