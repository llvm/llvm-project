; RUN: llc -O2 -mtriple=hexagon < %s
; REQUIRES: asserts

; Function Attrs: nounwind
define void @f0(i32 %a0) #0 {
b0:
  %v0 = icmp ugt i32 %a0, 1
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v1 = phi i32 [ %v2, %b1 ], [ 0, %b0 ]
  %v2 = add nsw i32 %v1, 2
  %v3 = icmp slt i32 %v2, 0
  br i1 %v3, label %b1, label %b2

b2:                                               ; preds = %b1, %b0
  unreachable
}

attributes #0 = { nounwind }
