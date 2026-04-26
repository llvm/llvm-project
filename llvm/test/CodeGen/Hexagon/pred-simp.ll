; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s
; CHECK-NOT: not(

target triple = "hexagon"

; Function Attrs: nounwind readnone
define i32 @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = icmp slt i32 %a0, %a1
  %v1 = add nsw i32 %a1, %a0
  %v2 = icmp sgt i32 %v1, 10
  %v3 = icmp eq i1 %v0, false
  %v4 = or i1 %v3, %v2
  br i1 %v4, label %b2, label %b1

b1:                                               ; preds = %b0
  %v5 = mul nsw i32 %a0, 2
  %v6 = icmp sgt i32 %v5, %a1
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v7 = phi i1 [ %v6, %b1 ], [ true, %b0 ]
  %v8 = zext i1 %v7 to i32
  ret i32 %v8
}

attributes #0 = { nounwind readnone }
