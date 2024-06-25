; RUN: llc < %s -mtriple=aarch64 -mattr=+v8.2a,+neon,+fullfp16  | FileCheck %s

define dso_local half @t_vfmah_f16(half %a, half %b, half %c) {
; CHECK-LABEL: t_vfmah_f16:
; CHECK:         fmadd h0, h1, h2, h0
; CHECK-NEXT:    ret
entry:
  %0 = tail call half @llvm.fma.f16(half %b, half %c, half %a)
  ret half %0
}

define half @fnma16(half %a, half %b, half %c) nounwind readnone ssp {
; CHECK-LABEL: fnma16:
; CHECK: fnmadd h0, h0, h1, h2
entry:
  %0 = tail call half @llvm.fma.f16(half %a, half %b, half %c)
  %mul = fmul half %0, -1.000000e+00
  ret half %mul
}

define half @fms16(half %a, half %b, half %c) nounwind readnone ssp {
; CHECK-LABEL: fms16:
; CHECK: fmsub h0, h0, h1, h2
entry:
  %mul = fneg half %b
  %0 = tail call half @llvm.fma.f16(half %a, half %mul, half %c)
  ret half %0
}

define half @fms16_com(half %a, half %b, half %c) nounwind readnone ssp {
; CHECK-LABEL: fms16_com:
; CHECK:       fmsub h0, h1, h0, h2
; CHECK-NEXT:  ret
entry:
  %mul = fneg half %b
  %0 = tail call half @llvm.fma.f16(half %mul, half %a, half %c)
  ret half %0
}

define half @fnms16(half %a, half %b, half %c) nounwind readnone ssp {
; CHECK-LABEL: fnms16:
; CHECK:       fnmsub h0, h0, h1, h2
; CHECK-NEXT:  ret
entry:
  %mul = fneg half %c
  %0 = tail call half @llvm.fma.f16(half %a, half %b, half %mul)
  ret half %0
}

define half @test_fmsub(half %a, half %b, half %c) {
; CHECK-LABEL: test_fmsub:
; CHECK:       fmsub h0, h0, h1, h2
; CHECK-NEXT:  ret
entry:
  %nega = fsub half -0.0, %a
  %val = call half @llvm.fma.f16(half %nega, half %b, half %c)
  ret half %val
}

define half @test_fnmadd(half %a, half %b, half %c) {
; CHECK-LABEL: test_fnmadd:
; CHECK:       fnmadd h0, h0, h1, h2
; CHECK-NEXT:  ret
entry:
  %nega = fsub half -0.0, %a
  %negc = fsub half -0.0, %c
  %val = call half @llvm.fma.f16(half %nega, half %b, half %negc)
  ret half %val
}

define half @test_fmadd(half %a, half %b, half %c) {
; CHECK-LABEL: test_fmadd:
; CHECK:       fmadd h0, h0, h1, h2
; CHECK-NEXT:  ret
entry:
  %nega = fsub half -0.0, %a
  %negb = fsub half -0.0, %b
  %val = call half @llvm.fma.f16(half %nega, half %negb, half %c)
  ret half %val
}

define half @test_fnmsub(half %a, half %b, half %c) {
; CHECK-LABEL: test_fnmsub:
; CHECK:       fnmsub h0, h0, h1, h2
; CHECK-NEXT:  ret
entry:
  %negc = fsub half -0.0, %c
  %val = call half @llvm.fma.f16(half %a, half %b, half %negc)
  ret half %val
}

declare half @llvm.fma.f16(half, half, half)
