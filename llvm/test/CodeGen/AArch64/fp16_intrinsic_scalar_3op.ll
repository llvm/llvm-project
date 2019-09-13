; RUN: llc < %s -mtriple=aarch64-eabi -mattr=+v8.2a,+neon,+fullfp16  | FileCheck %s

define dso_local half @t_vfmah_f16(half %a, half %b, half %c) {
; CHECK-LABEL: t_vfmah_f16:
; CHECK:         fmadd h0, h1, h2, h0
; CHECK-NEXT:    ret
entry:
  %0 = tail call half @llvm.fma.f16(half %b, half %c, half %a)
  ret half %0
}

define half @fnma16(half %a, half %b, half %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fnma16:
; CHECK: fnmadd h0, h0, h1, h2
  %0 = tail call half @llvm.fma.f16(half %a, half %b, half %c)
  %mul = fmul half %0, -1.000000e+00
  ret half %mul
}

define half @fms16(half %a, half %b, half %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fms16:
; CHECK: fmsub h0, h0, h1, h2
  %mul = fmul half %b, -1.000000e+00
  %0 = tail call half @llvm.fma.f16(half %a, half %mul, half %c)
  ret half %0
}

define half @fms16_com(half %a, half %b, half %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fms16_com:

; FIXME:       This should be a fmsub.

; CHECK:       fneg  h1, h1
; CHECK-NEXT:  fmadd h0, h1, h0, h2
  %mul = fmul half %b, -1.000000e+00
  %0 = tail call half @llvm.fma.f16(half %mul, half %a, half %c)
  ret half %0
}

define half @fnms16(half %a, half %b, half %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: fnms16:
; CHECK: fnmsub h0, h0, h1, h2
  %mul = fmul half %c, -1.000000e+00
  %0 = tail call half @llvm.fma.f16(half %a, half %b, half %mul)
  ret half %0
}

declare half @llvm.fma.f16(half, half, half)

