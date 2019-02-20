; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

; CHECK-NOT: p1 =

define i32 @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = icmp slt i32 %a0, %a1
  br i1 %v0, label %b1, label %b2

b1:
  ret i32 0

b2:
  %v1 = icmp slt i32 %a1, 100
  %v2 = select i1 %v1, i32 123, i32 321
  ret i32 %v2
}
