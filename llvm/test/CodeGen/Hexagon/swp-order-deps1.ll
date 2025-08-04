; RUN: llc -mtriple=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Check that the dependences are order correctly, and the list can be
; updated when the instruction to insert has a def and use conflict.

; Function Attrs: nounwind
define fastcc void @f0() #0 {
b0:
  br i1 undef, label %b7, label %b1

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b4

b2:                                               ; preds = %b1
  %v0 = load i16, ptr undef, align 2
  br label %b5

b3:                                               ; preds = %b5
  br label %b4

b4:                                               ; preds = %b3, %b1
  %v1 = phi i16 [ %v11, %b3 ], [ 0, %b1 ]
  br i1 false, label %b7, label %b6

b5:                                               ; preds = %b5, %b2
  %v2 = phi i16 [ %v3, %b5 ], [ undef, %b2 ]
  %v3 = phi i16 [ 0, %b5 ], [ %v0, %b2 ]
  %v4 = phi i16 [ %v2, %b5 ], [ undef, %b2 ]
  %v5 = phi i16 [ %v11, %b5 ], [ 0, %b2 ]
  %v6 = phi i32 [ %v12, %b5 ], [ undef, %b2 ]
  %v7 = or i16 0, %v5
  %v8 = lshr i16 %v4, 8
  %v9 = or i16 %v8, %v7
  %v10 = or i16 0, %v9
  %v11 = or i16 0, %v10
  %v12 = add nsw i32 %v6, -32
  %v13 = icmp sgt i32 %v12, 31
  br i1 %v13, label %b5, label %b3

b6:                                               ; preds = %b4
  br label %b7

b7:                                               ; preds = %b6, %b4, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
