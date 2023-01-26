; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

; Function Attrs: nounwind optsize readonly
define void @f0() #0 align 2 {
b0:
  %v0 = load i32, ptr undef, align 8
  %v1 = zext i32 %v0 to i64
  %v2 = add nuw nsw i64 %v1, 63
  %v3 = lshr i64 %v2, 6
  %v4 = trunc i64 %v3 to i32
  br i1 undef, label %b3, label %b1

b1:                                               ; preds = %b0
  %v5 = add nsw i32 %v4, -1
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v6 = phi i32 [ %v5, %b1 ], [ %v7, %b2 ]
  %v7 = add i32 %v6, -1
  %v8 = icmp sgt i32 %v7, -1
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

attributes #0 = { nounwind optsize readonly }
