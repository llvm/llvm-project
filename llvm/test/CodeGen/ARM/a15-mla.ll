; RUN: llc < %s  -march=arm -float-abi=hard -mcpu=cortex-a15 -mattr=+neon,+neonfp | FileCheck %s

; This test checks that the VMLxForwarting feature is disabled for A15.
; CHECK: fun_a
define <4 x i32> @fun_a(<4 x i32> %x, <4 x i32> %y) nounwind{
  %1 = add <4 x i32> %x, %y
; CHECK-NOT: vmul
; CHECK: vmla
  %2 = mul <4 x i32> %1, %1
  %3 = add <4 x i32> %y, %2
  ret <4 x i32> %3
}
