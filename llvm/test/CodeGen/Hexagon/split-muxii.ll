; RUN: llc -march=hexagon -O2 -hexagon-expand-condsets=true -hexagon-gen-mux-threshold=4 < %s | FileCheck %s
; CHECK-NOT: mux(p

target triple = "hexagon"

define void @f0() #0 {
b0:
  %v0 = load i32, i32* null, align 4
  %v1 = icmp slt i32 undef, %v0
  %v2 = zext i1 %v1 to i32
  %v3 = icmp sgt i32 undef, 0
  %v4 = zext i1 %v3 to i32
  %v5 = add nsw i32 %v2, %v4
  store i32 %v5, i32* undef, align 4
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b1, %b0
  unreachable
}

attributes #0 = { nounwind }
