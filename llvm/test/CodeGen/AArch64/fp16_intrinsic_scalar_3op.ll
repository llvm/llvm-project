; RUN: llc < %s -mtriple=aarch64-eabi -mattr=+v8.2a,+fullfp16  | FileCheck %s

declare half @llvm.fma.f16(half, half, half)

define dso_local half @t_vfmah_f16(half %a, half %b, half %c) {
; CHECK-LABEL: t_vfmah_f16:
; CHECK:         fmadd h0, h1, h2, h0
; CHECK-NEXT:    ret
entry:
  %0 = tail call half @llvm.fma.f16(half %b, half %c, half %a)
  ret half %0
}

