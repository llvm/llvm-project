; RUN: llc < %s -mtriple=arm-none-eabi -mattr=+v8.2a,+fullfp16,+neon  -float-abi=hard   | FileCheck %s --check-prefixes=CHECK,CHECK-HARD
; RUN: llc < %s -mtriple=arm-none-eabi -mattr=+v8.2a,+fullfp16,+neon  | FileCheck %s --check-prefixes=CHECK,CHECK-SOFTFP

declare <4 x half> @llvm.arm.neon.vpadd.v4f16(<4 x half>, <4 x half>)

define dso_local <4 x half> @t_vpadd_f16(<4 x half> %a, <4 x half> %b) {
; CHECK:   t_vpadd_f16:

; CHECK-HARD:       vpadd.f16 d0, d0, d1
; CHECK-HARD-NEXT:  bx  lr

; CHECK-SOFTFP:     vmov  [[D1:d[0-9]+]], r2, r3
; CHECK-SOFTFP:     vmov  [[D2:d[0-9]+]], r0, r1
; CHECK-SOFTFP:     vpadd.f16 [[D3:d[0-9]+]], [[D2]], [[D1]]
; CHECK-SOFTFP:     vmov  r0, r1, [[D3]]
; CHECK-SOFTFP:     bx  lr

entry:
  %vpadd_v2.i = tail call <4 x half> @llvm.arm.neon.vpadd.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vpadd_v2.i
}
