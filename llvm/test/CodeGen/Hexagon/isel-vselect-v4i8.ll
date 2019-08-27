; RUN: llc -march=hexagon < %s | FileCheck %s
; This used to crash with "cannot select (v4i8 vselect ...)"
; CHECK: vtrunehb

define <4 x i8> @f0(<4 x i8> %a0, <4 x i8> %a1) #0 {
  %v0 = icmp slt <4 x i8> %a0, %a1
  %v1 = select <4 x i1> %v0, <4 x i8> %a0, <4 x i8> %a1
  ret <4 x i8> %v1
}
