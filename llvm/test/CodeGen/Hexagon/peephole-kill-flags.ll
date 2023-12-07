; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s
; CHECK: memw

; Check that the testcase compiles without errors.

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0(ptr %a0, i32 %a1) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  %v0 = load i32, ptr %a0, align 4
  %v1 = mul nsw i32 2, %v0
  %v2 = icmp slt i32 %a1, %v1
  br i1 %v2, label %b2, label %b3

b2:                                               ; preds = %b1
  ret i32 0

b3:                                               ; preds = %b1
  ret i32 %v1
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
