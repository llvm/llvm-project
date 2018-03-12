; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: tstbit

; Function Attrs: nounwind readnone
define i32 @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = shl i32 1, %a1
  %v1 = and i32 %v0, %a0
  %v2 = icmp ne i32 %v1, 0
  %v3 = zext i1 %v2 to i32
  ret i32 %v3
}

attributes #0 = { nounwind readnone }
