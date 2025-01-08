; RUN: llc < %s -mtriple=aarch64 -mattr=+v8.2a,+fullfp16  | FileCheck %s

declare <4 x half> @llvm.aarch64.neon.fmulx.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.aarch64.neon.fmulx.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.aarch64.neon.fminnmp.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.aarch64.neon.fminnmp.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.aarch64.neon.fmaxnmp.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.aarch64.neon.fmaxnmp.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.aarch64.neon.fabd.v4f16(<4 x half>, <4 x half>)
declare <8 x half> @llvm.aarch64.neon.fabd.v8f16(<8 x half>, <8 x half>)
declare <4 x half> @llvm.fabs.v4f16(<4 x half>)
declare <8 x half> @llvm.fabs.v8f16(<8 x half>)

define dso_local <4 x half> @t_vdiv_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: t_vdiv_f16:
; CHECK:         fdiv v0.4h, v0.4h, v1.4h
; CHECK-NEXT:    ret
entry:
  %div.i = fdiv <4 x half> %a, %b
  ret <4 x half> %div.i
}

define dso_local <8 x half> @t_vdivq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: t_vdivq_f16:
; CHECK:         fdiv v0.8h, v0.8h, v1.8h
; CHECK-NEXT:    ret
entry:
  %div.i = fdiv <8 x half> %a, %b
  ret <8 x half> %div.i
}

define dso_local <4 x half> @t_vmulx_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: t_vmulx_f16:
; CHECK:         fmulx v0.4h, v0.4h, v1.4h
; CHECK-NEXT:    ret
entry:
  %vmulx2.i = tail call <4 x half> @llvm.aarch64.neon.fmulx.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vmulx2.i
}

define dso_local <8 x half> @t_vmulxq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: t_vmulxq_f16:
; CHECK:         fmulx v0.8h, v0.8h, v1.8h
; CHECK-NEXT:    ret
entry:
  %vmulx2.i = tail call <8 x half> @llvm.aarch64.neon.fmulx.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vmulx2.i
}

define dso_local <4 x half> @t_vpminnm_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: t_vpminnm_f16:
; CHECK:         fminnmp v0.4h, v0.4h, v1.4h
; CHECK-NEXT:    ret
entry:
  %vpminnm2.i = tail call <4 x half> @llvm.aarch64.neon.fminnmp.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vpminnm2.i
}

define dso_local <8 x half> @t_vpminnmq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: t_vpminnmq_f16:
; CHECK:         fminnmp v0.8h, v0.8h, v1.8h
; CHECK-NEXT:    ret
entry:
  %vpminnm2.i = tail call <8 x half> @llvm.aarch64.neon.fminnmp.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vpminnm2.i
}

define dso_local <4 x half> @t_vpmaxnm_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: t_vpmaxnm_f16:
; CHECK:         fmaxnmp v0.4h, v0.4h, v1.4h
; CHECK-NEXT:    ret
entry:
  %vpmaxnm2.i = tail call <4 x half> @llvm.aarch64.neon.fmaxnmp.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vpmaxnm2.i
}

define dso_local <8 x half> @t_vpmaxnmq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: t_vpmaxnmq_f16:
; CHECK:         fmaxnmp v0.8h, v0.8h, v1.8h
; CHECK-NEXT:    ret
entry:
  %vpmaxnm2.i = tail call <8 x half> @llvm.aarch64.neon.fmaxnmp.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vpmaxnm2.i
}

define dso_local <4 x half> @t_vabd_f16(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: t_vabd_f16:
; CHECK:         fabd v0.4h, v0.4h, v1.4h
; CHECK-NEXT:    ret
entry:
  %vabdh_f16 = tail call <4 x half> @llvm.aarch64.neon.fabd.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %vabdh_f16
}

define dso_local <8 x half> @t_vabdq_f16(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: t_vabdq_f16:
; CHECK:         fabd v0.8h, v0.8h, v1.8h
; CHECK-NEXT:    ret
entry:
  %vabdh_f16 = tail call <8 x half> @llvm.aarch64.neon.fabd.v8f16(<8 x half> %a, <8 x half> %b)
  ret <8 x half> %vabdh_f16
}

define dso_local <4 x half> @t_vabd_f16_from_fsub_fabs(<4 x half> %a, <4 x half> %b) {
; CHECK-LABEL: t_vabd_f16_from_fsub_fabs:
; CHECK:         fabd v0.4h, v0.4h, v1.4h
; CHECK-NEXT:    ret
entry:
  %sub = fsub <4 x half> %a, %b
  %abs = tail call <4 x half> @llvm.fabs.v4f16(<4 x half> %sub)
  ret <4 x half> %abs
}

define dso_local <8 x half> @t_vabdq_f16_from_fsub_fabs(<8 x half> %a, <8 x half> %b) {
; CHECK-LABEL: t_vabdq_f16_from_fsub_fabs:
; CHECK:         fabd v0.8h, v0.8h, v1.8h
; CHECK-NEXT:    ret
entry:
  %sub = fsub <8 x half> %a, %b
  %abs = tail call <8 x half> @llvm.fabs.v8f16(<8 x half> %sub)
  ret <8 x half> %abs
}
