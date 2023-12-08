; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that the hardware loop pass does not alter the comparison
; to use the result from the induction expression instead of
; from the Phi.

; CHECK: cmpb.gtu([[REG0:r[0-9]+]]
; CHECK: [[REG0]] = add([[REG0]],

define void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b1, label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ %v3, %b2 ], [ undef, %b1 ]
  %v1 = trunc i32 %v0 to i8
  %v2 = icmp ugt i8 %v1, 44
  %v3 = add i32 %v0, -30
  br i1 %v2, label %b2, label %b3

b3:                                               ; preds = %b2
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
