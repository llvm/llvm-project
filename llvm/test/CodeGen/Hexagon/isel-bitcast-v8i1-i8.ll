; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Check that this doesn't crash.
; CHECK-LABEL: foo:
; CHECK: p[[P:[0-3]]] = vcmpb.eq
; CHECK: r[[R:[0-9]+]] = p[[P]]
; CHECK: and(r[[R]],#32)

define i32 @foo(<8 x i8> %a0, <8 x i8> %a1) #0 {
  %v0 = icmp eq <8 x i8> %a0, %a1
  %v1 = bitcast <8 x i1> %v0 to i8
  %v2 = and i8 %v1, 32
  %v3 = zext i8 %v2 to i32
  ret i32 %v3
}

attributes #0 = { readnone nounwind }
