; RUN: llc -mtriple=powerpc-unknown-linux-gnu -mattr=+vsx -verify-machineinstrs < %s | FileCheck %s --check-prefix=BE-VSX
; RUN: llc -mtriple=powerpcle-unknown-linux-gnu -mattr=+vsx -verify-machineinstrs < %s | FileCheck %s --check-prefix=LE-VSX
; RUN: llc -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs < %s | FileCheck %s --check-prefix=P8

; Check that 128-bit scalar and single-element vector arguments assigned to
; AltiVec registers can be lowered for the 32-bit SVR4 ABI.

define void @store_fp128(fp128 %x, ptr %p) {
; BE-VSX-LABEL:     store_fp128:
; BE-VSX:           stxvw4x 34,
; LE-VSX-LABEL:     store_fp128:
; LE-VSX:           xxswapd
; LE-VSX:           stxvd2x
  store fp128 %x, ptr %p, align 16
  ret void
}

define void @store_v1i128(<1 x i128> %x, ptr %p) {
; P8-LABEL: store_v1i128:
; P8:       stxvw4x 34,
  store <1 x i128> %x, ptr %p, align 16
  ret void
}
