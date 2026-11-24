; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s

; Returning an integer in w0 must not reset the unrelated physical register
; whose MCRegister ID equals RegUnit(W0) (AArch64::S30), leaving v30 unzeroed.
define i32 @ret_w0_uses_s30(i32 %x, ptr %p) nounwind "zero-call-used-regs"="used" {
; CHECK-LABEL: ret_w0_uses_s30:
; CHECK:       ldr s30, [x1]
; CHECK:       str s30, [x1]
; CHECK:       movi v30.2d, #0000000000000000
; CHECK:       ret
entry:
  %f0  = load volatile float, ptr %p
  %f1  = load volatile float, ptr %p
  %f2  = load volatile float, ptr %p
  %f3  = load volatile float, ptr %p
  %f4  = load volatile float, ptr %p
  %f5  = load volatile float, ptr %p
  %f6  = load volatile float, ptr %p
  %f7  = load volatile float, ptr %p
  %f8  = load volatile float, ptr %p
  %f9  = load volatile float, ptr %p
  %f10 = load volatile float, ptr %p
  %f11 = load volatile float, ptr %p
  %f12 = load volatile float, ptr %p
  %f13 = load volatile float, ptr %p
  %f14 = load volatile float, ptr %p
  %f15 = load volatile float, ptr %p
  %f16 = load volatile float, ptr %p
  %f17 = load volatile float, ptr %p
  %f18 = load volatile float, ptr %p
  %f19 = load volatile float, ptr %p
  %f20 = load volatile float, ptr %p
  %f21 = load volatile float, ptr %p
  %f22 = load volatile float, ptr %p
  %f23 = load volatile float, ptr %p
  store volatile float %f0, ptr %p
  store volatile float %f1, ptr %p
  store volatile float %f2, ptr %p
  store volatile float %f3, ptr %p
  store volatile float %f4, ptr %p
  store volatile float %f5, ptr %p
  store volatile float %f6, ptr %p
  store volatile float %f7, ptr %p
  store volatile float %f8, ptr %p
  store volatile float %f9, ptr %p
  store volatile float %f10, ptr %p
  store volatile float %f11, ptr %p
  store volatile float %f12, ptr %p
  store volatile float %f13, ptr %p
  store volatile float %f14, ptr %p
  store volatile float %f15, ptr %p
  store volatile float %f16, ptr %p
  store volatile float %f17, ptr %p
  store volatile float %f18, ptr %p
  store volatile float %f19, ptr %p
  store volatile float %f20, ptr %p
  store volatile float %f21, ptr %p
  store volatile float %f22, ptr %p
  store volatile float %f23, ptr %p
  ret i32 %x
}
