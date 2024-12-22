; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; Make sure no mux with 0 is generated.
; CHECK-NOT: mux{{.*}}#0
; CHECK: endloop

target triple = "hexagon"

; Function Attrs: nounwind readnone
define i32 @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = icmp ugt i32 %a0, %a1
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi i32 [ 0, %b0 ], [ %v5, %b1 ]
  %v2 = phi i32 [ 0, %b0 ], [ %v7, %b1 ]
  %v3 = phi i32 [ 1, %b0 ], [ %v6, %b1 ]
  %v4 = select i1 %v0, i32 %v3, i32 0
  %v5 = or i32 %v1, %v4
  %v6 = shl i32 %v3, 1
  %v7 = add i32 %v2, 1
  %v8 = icmp eq i32 %v7, 32
  br i1 %v8, label %b2, label %b1

b2:                                               ; preds = %b1
  ret i32 %v5
}

attributes #0 = { nounwind readnone }
