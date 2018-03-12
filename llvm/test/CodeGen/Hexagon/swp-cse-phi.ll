; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; This test checks that we don't assert when the Phi value from the
; loop is actually defined prior to the loop, e.g., from CSE.

define fastcc void @f0() {
b0:
  br i1 undef, label %b10, label %b1

b1:                                               ; preds = %b0
  br i1 undef, label %b3, label %b2

b2:                                               ; preds = %b1
  br label %b8

b3:                                               ; preds = %b1
  br i1 undef, label %b4, label %b6

b4:                                               ; preds = %b3
  %v0 = load i16, i16* undef, align 2
  br label %b7

b5:                                               ; preds = %b7
  br label %b6

b6:                                               ; preds = %b5, %b3
  %v1 = phi i16 [ %v9, %b5 ], [ 0, %b3 ]
  br i1 undef, label %b10, label %b9

b7:                                               ; preds = %b7, %b4
  %v2 = phi i16 [ 0, %b7 ], [ %v0, %b4 ]
  %v3 = phi i16 [ %v9, %b7 ], [ 0, %b4 ]
  %v4 = phi i32 [ %v10, %b7 ], [ undef, %b4 ]
  %v5 = or i16 0, %v3
  %v6 = or i16 0, %v5
  %v7 = or i16 0, %v6
  %v8 = lshr i16 %v2, 8
  %v9 = or i16 %v8, %v7
  %v10 = add nsw i32 %v4, -32
  %v11 = icmp sgt i32 %v10, 31
  br i1 %v11, label %b7, label %b5

b8:                                               ; preds = %b8, %b2
  br label %b8

b9:                                               ; preds = %b6
  br label %b10

b10:                                              ; preds = %b9, %b6, %b0
  ret void
}
