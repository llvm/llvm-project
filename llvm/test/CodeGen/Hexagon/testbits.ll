; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: p0 = bitsset(r0,r1)
define i32 @f0(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = and i32 %a0, %a1
  %v1 = icmp eq i32 %v0, %a1
  %v2 = select i1 %v1, i32 2, i32 3
  ret i32 %v2
}

; CHECK-LABEL: f1:
; CHECK: p0 = bitsclr(r0,r1)
define i32 @f1(i32 %a0, i32 %a1) #0 {
b0:
  %v0 = and i32 %a0, %a1
  %v1 = icmp eq i32 %v0, 0
  %v2 = select i1 %v1, i32 2, i32 3
  ret i32 %v2
}

; CHECK-LABEL: f2:
; CHECK: p0 = bitsclr(r0,#37)
define i32 @f2(i32 %a0) #0 {
b0:
  %v0 = and i32 %a0, 37
  %v1 = icmp eq i32 %v0, 0
  %v2 = select i1 %v1, i32 2, i32 3
  ret i32 %v2
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
