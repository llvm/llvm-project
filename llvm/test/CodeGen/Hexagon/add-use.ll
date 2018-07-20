; RUN: llc -march=hexagon < %s | FileCheck %s

; Do not want to see register copies in the loop.
; CHECK-NOT: r{{[0-9]*}} = r{{[0-9]*}}

target triple = "hexagon"

; Function Attrs: nounwind readnone
define i32 @f0(i32 %a0) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v1, %b1 ]
  %v1 = add nsw i32 %v0, 3
  %v2 = add i32 %v1, -3
  %v3 = icmp slt i32 %v2, %a0
  br i1 %v3, label %b1, label %b2

b2:                                               ; preds = %b1
  %v4 = phi i32 [ %v1, %b1 ]
  %v5 = icmp slt i32 %v4, 100
  %v6 = zext i1 %v5 to i32
  ret i32 %v6
}

attributes #0 = { nounwind readnone }
