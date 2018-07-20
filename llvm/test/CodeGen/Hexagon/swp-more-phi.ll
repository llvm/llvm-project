; RUN: llc -march=hexagon --enable-pipeliner -hexagon-expand-condsets=0 < %s
; REQUIRES: asserts

; Disable expand-condsets because it will assert on undefined registers.

define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b3, %b2
  br i1 undef, label %b4, label %b3

b4:                                               ; preds = %b3, %b2
  %v0 = ashr i32 undef, 25
  %v1 = mul nsw i32 %v0, 2
  %v2 = load i8, i8* undef, align 1
  br i1 undef, label %b5, label %b10

b5:                                               ; preds = %b4
  br i1 undef, label %b6, label %b9

b6:                                               ; preds = %b5
  br label %b7

b7:                                               ; preds = %b7, %b6
  br i1 undef, label %b7, label %b8

b8:                                               ; preds = %b7
  br i1 undef, label %b10, label %b9

b9:                                               ; preds = %b9, %b8, %b5
  %v3 = phi i8 [ %v7, %b9 ], [ undef, %b8 ], [ %v2, %b5 ]
  %v4 = phi i32 [ %v8, %b9 ], [ undef, %b8 ], [ 1, %b5 ]
  %v5 = add i32 %v4, undef
  %v6 = load i8, i8* undef, align 1
  %v7 = select i1 undef, i8 %v6, i8 %v3
  %v8 = add nsw i32 %v4, 1
  %v9 = icmp eq i32 %v8, %v1
  br i1 %v9, label %b10, label %b9

b10:                                              ; preds = %b9, %b8, %b4
  %v10 = phi i8 [ %v2, %b4 ], [ undef, %b8 ], [ %v7, %b9 ]
  br i1 false, label %b11, label %b12

b11:                                              ; preds = %b10
  unreachable

b12:                                              ; preds = %b10
  br label %b13

b13:                                              ; preds = %b13, %b12
  br label %b13
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
