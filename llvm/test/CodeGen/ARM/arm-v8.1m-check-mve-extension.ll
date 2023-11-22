; RUN: llc -mtriple=arm-none-eabi -mcpu=cortex-m85 --float-abi=hard %s -o - | FileCheck %s
; RUN: llc -mtriple=arm-none-eabi -mcpu=cortex-m55 --float-abi=hard %s -o - | FileCheck %s

; CHECK: .fpu   fpv5-d16
; CHECK-NEXT:  .arch_extension mve.fp

define <4 x float> @vsubf32(<4 x float> %A, <4 x float> %B) {
; CHECK-LABEL: vsubf32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vsub.f32 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fsub <4 x float> %A, %B
  ret <4 x float> %tmp3
}
