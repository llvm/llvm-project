; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Check that the verifier doesn't fail due to incorrect
; ordering of registers caused by PHI elimination.

; Function Attrs: readnone
define i32 @f0(i32 %a0, i32 %a1, i32 %a2) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %a1, %b0 ], [ %v2, %b1 ]
  %v1 = phi i32 [ 0, %b0 ], [ %v4, %b1 ]
  %v2 = phi i32 [ %a0, %b0 ], [ %v0, %b1 ]
  %v3 = icmp slt i32 %v1, %a2
  %v4 = add nsw i32 %v1, 1
  br i1 %v3, label %b1, label %b2

b2:                                               ; preds = %b1
  %v5 = add nsw i32 %v2, %v0
  ret i32 %v5
}

attributes #0 = { readnone }
