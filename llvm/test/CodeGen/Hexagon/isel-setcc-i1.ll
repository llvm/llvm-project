; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s

; Check that this compiles successfully.
; CHECK: if (p0)

target triple = "hexagon"

define void @fred() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v2 = load i32, i32* undef, align 4
  %v3 = select i1 undef, i32 %v2, i32 0
  %v4 = and i32 %v3, 7
  %v5 = icmp eq i32 %v4, 4
  %v6 = or i1 undef, %v5
  %v7 = and i1 undef, %v6
  %v8 = xor i1 %v7, true
  %v9 = or i1 undef, %v8
  br i1 %v9, label %b10, label %b1

b10:                                              ; preds = %b1
  unreachable
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
