; RUN: llc < %s -mtriple=aarch64 -mattr=+v8.2a,+fullfp16  | FileCheck %s

declare <4 x half> @llvm.nearbyint.v4f16(<4 x half>)
declare <8 x half> @llvm.nearbyint.v8f16(<8 x half>)
declare <4 x half> @llvm.sqrt.v4f16(<4 x half>)
declare <8 x half> @llvm.sqrt.v8f16(<8 x half>)

define dso_local <4 x half> @t_vrndi_f16(<4 x half> %a) {
; CHECK-LABEL: t_vrndi_f16:
; CHECK:         frinti v0.4h, v0.4h
; CHECK-NEXT:    ret
entry:
  %vrndi1.i = tail call <4 x half> @llvm.nearbyint.v4f16(<4 x half> %a)
  ret <4 x half> %vrndi1.i
}

define dso_local <8 x half> @t_vrndiq_f16(<8 x half> %a) {
; CHECK-LABEL: t_vrndiq_f16:
; CHECK:         frinti v0.8h, v0.8h
; CHECK-NEXT:    ret
entry:
  %vrndi1.i = tail call <8 x half> @llvm.nearbyint.v8f16(<8 x half> %a)
  ret <8 x half> %vrndi1.i
}

define dso_local <4 x half> @t_vsqrt_f16(<4 x half> %a) {
; CHECK-LABEL: t_vsqrt_f16:
; CHECK:         fsqrt v0.4h, v0.4h
; CHECK-NEXT:    ret
entry:
  %vsqrt.i = tail call <4 x half> @llvm.sqrt.v4f16(<4 x half> %a)
  ret <4 x half> %vsqrt.i
}

define dso_local <8 x half> @t_vsqrtq_f16(<8 x half> %a) {
; CHECK-LABEL: t_vsqrtq_f16:
; CHECK:         fsqrt v0.8h, v0.8h
; CHECK-NEXT:    ret
entry:
  %vsqrt.i = tail call <8 x half> @llvm.sqrt.v8f16(<8 x half> %a)
  ret <8 x half> %vsqrt.i
}
