; RUN: llc -mtriple=arm-eabi -float-abi=hard -mcpu=cortex-a15 -mattr=+neon,+neonfp %s -o - \
; RUN:  | FileCheck %s

; This test checks that the VMLxForwarting feature is disabled for A15.
; CHECK: fun_a:
define <4 x i32> @fun_a(<4 x i32> %x, <4 x i32> %y) nounwind{
  %1 = add <4 x i32> %x, %y
; CHECK-NOT: vmul
; CHECK: vmla
  %2 = mul <4 x i32> %1, %1
  %3 = add <4 x i32> %y, %2
  ret <4 x i32> %3
}

; This tests checks that VMLA FP patterns can be matched in instruction selection when targeting
; Cortex-A15.
; CHECK: fun_b:
define <4 x float> @fun_b(<4 x float> %x, <4 x float> %y, <4 x float> %z) nounwind{
; CHECK: vmla.f32
  %t = fmul <4 x float> %x, %y
  %r = fadd <4 x float> %t, %z
  ret <4 x float> %r
}

; This tests checks that FP VMLA instructions are not expanded into separate multiply/addition
; operations when targeting Cortex-A15.
; CHECK: fun_c:
define <4 x float> @fun_c(<4 x float> %x, <4 x float> %y, <4 x float> %z, <4 x float> %u, <4 x float> %v) nounwind{
; CHECK: vmla.f32
  %t1 = fmul <4 x float> %x, %y
  %r1 = fadd <4 x float> %t1, %z
; CHECK: vmla.f32
  %t2 = fmul <4 x float> %u, %v
  %r2 = fadd <4 x float> %t2, %r1
  ret <4 x float> %r2
}

