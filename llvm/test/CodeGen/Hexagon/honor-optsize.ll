; RUN: llc -mtriple=hexagon -O3 < %s | FileCheck %s

target triple = "hexagon"

; CHECK: f0:
; CHECK:   call __save_r16_through_r21
; CHECK:   .size	f0
define i32 @f0(ptr nocapture %a0) #0 {
b0:
  %v0 = tail call i32 @f1() #0
  %v1 = tail call i32 @f1() #0
  %v2 = tail call i32 @f1() #0
  %v3 = tail call i32 @f1() #0
  %v4 = load i8, ptr %a0, align 1
  %v5 = icmp eq i8 %v4, 0
  br i1 %v5, label %b4, label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v6 = phi i32 [ %v10, %b2 ], [ 0, %b1 ]
  %v7 = phi i32 [ %v2, %b2 ], [ %v1, %b1 ]
  %v8 = phi i32 [ %v7, %b2 ], [ %v0, %b1 ]
  %v9 = tail call i32 @f1() #0
  %v10 = add nsw i32 %v6, %v8
  %v11 = tail call i32 @f1() #0
  %v12 = load i8, ptr %a0, align 1
  %v13 = icmp eq i8 %v12, 0
  br i1 %v13, label %b3, label %b2

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v14 = phi i32 [ 0, %b0 ], [ %v10, %b3 ]
  ret i32 %v14
}

; Function Attrs: nounwind optsize
declare i32 @f1(...) #0

attributes #0 = { nounwind optsize }
