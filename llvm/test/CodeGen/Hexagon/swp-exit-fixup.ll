; RUN: llc -mtriple=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Make sure we fix up the Phis when we connect the last
; epilog block to the CFG.

define void @f0(ptr nocapture %a0) #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  br label %b3

b2:                                               ; preds = %b0
  unreachable

b3:                                               ; preds = %b3, %b1
  br i1 undef, label %b4, label %b3

b4:                                               ; preds = %b3
  br i1 undef, label %b6, label %b5

b5:                                               ; preds = %b4
  store i16 4096, ptr %a0, align 2
  br label %b11

b6:                                               ; preds = %b4
  br i1 undef, label %b7, label %b8

b7:                                               ; preds = %b7, %b6
  br label %b7

b8:                                               ; preds = %b8, %b6
  br i1 undef, label %b9, label %b8

b9:                                               ; preds = %b8
  %v0 = icmp sgt i32 undef, 1
  br i1 %v0, label %b10, label %b11

b10:                                              ; preds = %b10, %b9
  %v1 = phi i32 [ %v8, %b10 ], [ 1, %b9 ]
  %v2 = getelementptr inbounds [11 x i32], ptr undef, i32 0, i32 %v1
  %v3 = load i32, ptr undef, align 4
  %v4 = add nsw i32 %v3, 0
  %v5 = add nsw i32 %v4, 2048
  %v6 = lshr i32 %v5, 12
  %v7 = trunc i32 %v6 to i16
  store i16 %v7, ptr undef, align 2
  %v8 = add nsw i32 %v1, 1
  %v9 = icmp eq i32 %v8, undef
  br i1 %v9, label %b11, label %b10

b11:                                              ; preds = %b10, %b9, %b5
  %v10 = phi i1 [ false, %b9 ], [ false, %b5 ], [ %v0, %b10 ]
  br i1 undef, label %b16, label %b12

b12:                                              ; preds = %b11
  br i1 undef, label %b13, label %b16

b13:                                              ; preds = %b12
  br i1 %v10, label %b14, label %b15

b14:                                              ; preds = %b14, %b13
  br i1 undef, label %b15, label %b14

b15:                                              ; preds = %b14, %b13
  br label %b16

b16:                                              ; preds = %b15, %b12, %b11
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
