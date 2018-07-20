; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Make sure this doesn't crash.
; CHECK: jumpr r31

target triple = "hexagon"

; Function Attrs: nounwind
declare void @f0() #0

; Function Attrs: nounwind
define i32 @f1(i32 %a0) #0 {
b0:
  %v0 = icmp slt i32 %a0, 3
  %v1 = select i1 %v0, void ()* @f0, void ()* null
  %v2 = ptrtoint void ()* %v1 to i32
  ret i32 %v2
}

attributes #0 = { nounwind }
