; Test fp16 -> s16 conversion intrinsics which require special handling to ensure correct behaviour.
; RUN: llc < %s -mtriple=aarch64 -global-isel=0 -mattr=+v8.2a,+fullfp16  | FileCheck %s --check-prefixes=CHECK-SD

declare i16 @llvm.aarch64.neon.fcvtzs.i16.f16(half)

define i16 @fcvtzs_intrinsic_i16(half %a) {
; CHECK-SD-LABEL: fcvtzs_intrinsic_i16:
; CHECK-SD:       // %bb.0: // %entry
; CHECK-SD-NEXT:    fcvtzs w8, h0
; CHECK-SD-NEXT:    mov w9, #32767
; CHECK-SD-NEXT:    cmp w8, w9
; CHECK-SD-NEXT:    csel w8, w8, w9, lt
; CHECK-SD-NEXT:    mov w9, #-32768
; CHECK-SD-NEXT:    cmn w8, #8, lsl #12
; CHECK-SD-NEXT:    csel
; CHECK-SD-NEXT:    ret
entry:
  %fcvt = tail call i16 @llvm.aarch64.neon.fcvtzs.i16.f16(half %a)
  ret i16 %fcvt
}
