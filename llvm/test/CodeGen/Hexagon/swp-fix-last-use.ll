; RUN: llc -march=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; We need to rename uses that occurs after the loop.

define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b5

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b4

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ %v1, %b2 ], [ 1, %b1 ]
  store i16 0, ptr undef, align 2
  store i16 0, ptr undef, align 2
  %v1 = add nsw i32 %v0, 4
  %v2 = icmp slt i32 %v1, undef
  br i1 %v2, label %b2, label %b3

b3:                                               ; preds = %b2
  %v3 = icmp eq i32 %v1, undef
  br i1 %v3, label %b5, label %b4

b4:                                               ; preds = %b4, %b3, %b1
  br i1 undef, label %b5, label %b4

b5:                                               ; preds = %b4, %b3, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
