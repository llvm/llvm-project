; RUN: llc -march=hexagon -enable-aa-sched-mi -enable-pipeliner < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b2
  unreachable

b4:                                               ; preds = %b2
  br i1 undef, label %b5, label %b6

b5:                                               ; preds = %b4
  unreachable

b6:                                               ; preds = %b4
  br label %b7

b7:                                               ; preds = %b7, %b6
  br i1 undef, label %b8, label %b7

b8:                                               ; preds = %b7
  br i1 undef, label %b15, label %b9

b9:                                               ; preds = %b8
  br label %b10

b10:                                              ; preds = %b10, %b9
  br i1 undef, label %b11, label %b10

b11:                                              ; preds = %b10
  br label %b12

b12:                                              ; preds = %b12, %b11
  br i1 undef, label %b13, label %b12

b13:                                              ; preds = %b13, %b12
  %v0 = phi i32 [ %v5, %b13 ], [ 0, %b12 ]
  %v1 = getelementptr inbounds [11 x i32], ptr undef, i32 0, i32 %v0
  %v2 = load i32, ptr %v1, align 4
  %v3 = add i32 %v2, 1
  %v4 = lshr i32 %v3, 1
  store i32 %v4, ptr %v1, align 4
  store i32 0, ptr %v1, align 4
  %v5 = add nsw i32 %v0, 1
  %v6 = icmp eq i32 %v5, 11
  br i1 %v6, label %b14, label %b13

b14:                                              ; preds = %b13
  br label %b15

b15:                                              ; preds = %b14, %b8
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
