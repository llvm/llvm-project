; RUN: llc -march=hexagon -O2 -hexagon-expand-condsets=0 -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s

; CHECK: mux
; CHECK: cmp{{.*\.new}}

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp ne i32 %a0, 0
  %v1 = add nsw i32 %a2, -1
  %v2 = select i1 %v0, i32 10, i32 %v1
  %v3 = icmp eq i32 %v2, %a2
  br i1 %v3, label %b1, label %b2

b1:                                               ; preds = %b0
  %v4 = shl nsw i32 %a2, 1
  %v5 = tail call i32 @f1(i32 %v4) #0
  br label %b3

b2:                                               ; preds = %b0
  %v6 = tail call i32 @f1(i32 %a0) #0
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v7 = phi i32 [ %v5, %b1 ], [ %v6, %b2 ]
  ret i32 %v7
}

declare i32 @f1(i32)

attributes #0 = { nounwind }
