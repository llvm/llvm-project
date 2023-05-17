; RUN: opt -march=hexagon -hexagon-loop-idiom -S < %s | FileCheck %s
; REQUIRES: asserts
;
; Check for sane output, this used to crash.
; CHECK: define void @fred

; The conversion of shifts from right to left failed, but the return
; code was not checked and the transformation proceeded.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@A = common global [256 x i32] zeroinitializer, align 8

; Function Attrs: noinline nounwind
define void @fred() local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b13, %b0
  %v2 = phi i32 [ 0, %b0 ], [ %v16, %b13 ]
  br label %b3

b3:                                               ; preds = %b3, %b1
  %v4 = phi i32 [ %v2, %b1 ], [ %v10, %b3 ]
  %v5 = phi i32 [ 0, %b1 ], [ %v11, %b3 ]
  %v6 = and i32 %v4, 1
  %v7 = icmp ne i32 %v6, 0
  %v8 = lshr i32 %v4, 1
  %v9 = xor i32 %v8, 123456789
  %v10 = select i1 %v7, i32 %v9, i32 %v8
  %v11 = add nuw nsw i32 %v5, 1
  %v12 = icmp ne i32 %v11, 8
  br i1 %v12, label %b3, label %b13

b13:                                              ; preds = %b3
  %v14 = phi i32 [ %v10, %b3 ]
  %v15 = getelementptr inbounds [256 x i32], ptr @A, i32 0, i32 %v2
  store i32 %v14, ptr %v15, align 4
  %v16 = add nuw nsw i32 %v2, 1
  %v17 = icmp ne i32 %v16, 256
  br i1 %v17, label %b1, label %b18

b18:                                              ; preds = %b13
  ret void
}

attributes #0 = { noinline nounwind "target-cpu"="hexagonv60" }
