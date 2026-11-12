; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that the hardware loop pass does not alter the comparison
; to use the result from the induction expression instead of
; from the Phi.

; CHECK: cmpb.gtu([[REG0:r[0-9]+]]
; CHECK: [[REG0]] = add([[REG0]],

define i32 @f0(i32 %init) #0 {
b0:
  br label %b2

b2:                                               ; preds = %b2, %b0
  %v0 = phi i32 [ %v3, %b2 ], [ %init, %b0 ]
  %v1 = trunc i32 %v0 to i8
  %v2 = icmp ugt i8 %v1, 44
  %v3 = add i32 %v0, -30
  br i1 %v2, label %b2, label %b3

b3:                                               ; preds = %b2
  ret i32 %v0
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
