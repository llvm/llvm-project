; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that we generate a hardware loop for long long counters.
; Tests signed/unsigned GT, EQ, and NEQ cases.

; signed GT case
; CHECK-LABEL: f0:
; CHECK: loop0
define i32 @f0(i32* nocapture %a0) #0 {
b0:
  br label %b1
b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v5, %b1 ]
  %v1 = phi i64 [ 0, %b0 ], [ %v6, %b1 ]
  %v2 = trunc i64 %v1 to i32
  %v3 = getelementptr inbounds i32, i32* %a0, i32 %v2
  %v4 = load i32, i32* %v3, align 4
  %v5 = add nsw i32 %v4, %v0
  %v6 = add nsw i64 %v1, 1
  %v7 = icmp slt i64 %v6, 8
  br i1 %v7, label %b1, label %b2

b2:                                               ; preds = %b1
  ret i32 %v5
}

; unsigned signed GT case
; CHECK-LABEL: f1:
; CHECK: loop0
define i32 @f1(i32* nocapture %a0) #0 {
b0:
  br label %b1
b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v5, %b1 ]
  %v1 = phi i64 [ 0, %b0 ], [ %v6, %b1 ]
  %v2 = trunc i64 %v1 to i32
  %v3 = getelementptr inbounds i32, i32* %a0, i32 %v2
  %v4 = load i32, i32* %v3, align 4
  %v5 = add nsw i32 %v4, %v0
  %v6 = add i64 %v1, 1
  %v7 = icmp ult i64 %v6, 8
  br i1 %v7, label %b1, label %b2

b2:                                               ; preds = %b1
  ret i32 %v5
}

; EQ case
; CHECK-LABEL: f2:
; CHECK: loop0
define i32 @f2(i32* nocapture %a0) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v5, %b1 ]
  %v1 = phi i64 [ 0, %b0 ], [ %v6, %b1 ]
  %v2 = trunc i64 %v1 to i32
  %v3 = getelementptr inbounds i32, i32* %a0, i32 %v2
  %v4 = load i32, i32* %v3, align 4
  %v5 = add nsw i32 %v4, %v0
  %v6 = add nsw i64 %v1, 1
  %v7 = icmp eq i64 %v6, 8
  br i1 %v7, label %b2, label %b1

b2:                                               ; preds = %b1
  ret i32 %v5
}

attributes #0 = { nounwind readonly "target-cpu"="hexagonv55" }
