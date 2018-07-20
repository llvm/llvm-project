; RUN: llc < %s -mtriple=aarch64-eabi -mattr=+v8.2a,+fullfp16  | FileCheck %s

declare <4 x half> @llvm.fma.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <8 x half> @llvm.fma.v8f16(<8 x half>, <8 x half>, <8 x half>)

define dso_local <4 x half> @t_vfma_f16(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; CHECK-LABEL: t_vfma_f16:
; CHECK:         fmla v0.4h, v2.4h, v1.4h
; CHECK-NEXT:    ret
entry:
  %0 = tail call <4 x half> @llvm.fma.v4f16(<4 x half> %b, <4 x half> %c, <4 x half> %a)
  ret <4 x half> %0
}

define dso_local <8 x half> @t_vfmaq_f16(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; CHECK-LABEL: t_vfmaq_f16:
; CHECK:         fmla v0.8h, v2.8h, v1.8h
; CHECK-NEXT:    ret
entry:
  %0 = tail call <8 x half> @llvm.fma.v8f16(<8 x half> %b, <8 x half> %c, <8 x half> %a)
  ret <8 x half> %0
}
