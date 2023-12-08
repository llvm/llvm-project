; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 | FileCheck %s

@var = global float 0.0

define void @foo() {
; CHECK-LABEL: foo:

; CHECK: stp d15, d14, [sp
; CHECK: stp d13, d12, [sp
; CHECK: stp d11, d10, [sp
; CHECK: stp d9, d8, [sp

  ; Create lots of live variables to exhaust the supply of
  ; caller-saved registers
  %val1 = load volatile float, ptr @var
  %val2 = load volatile float, ptr @var
  %val3 = load volatile float, ptr @var
  %val4 = load volatile float, ptr @var
  %val5 = load volatile float, ptr @var
  %val6 = load volatile float, ptr @var
  %val7 = load volatile float, ptr @var
  %val8 = load volatile float, ptr @var
  %val9 = load volatile float, ptr @var
  %val10 = load volatile float, ptr @var
  %val11 = load volatile float, ptr @var
  %val12 = load volatile float, ptr @var
  %val13 = load volatile float, ptr @var
  %val14 = load volatile float, ptr @var
  %val15 = load volatile float, ptr @var
  %val16 = load volatile float, ptr @var
  %val17 = load volatile float, ptr @var
  %val18 = load volatile float, ptr @var
  %val19 = load volatile float, ptr @var
  %val20 = load volatile float, ptr @var
  %val21 = load volatile float, ptr @var
  %val22 = load volatile float, ptr @var
  %val23 = load volatile float, ptr @var
  %val24 = load volatile float, ptr @var
  %val25 = load volatile float, ptr @var
  %val26 = load volatile float, ptr @var
  %val27 = load volatile float, ptr @var
  %val28 = load volatile float, ptr @var
  %val29 = load volatile float, ptr @var
  %val30 = load volatile float, ptr @var
  %val31 = load volatile float, ptr @var
  %val32 = load volatile float, ptr @var

  store volatile float %val1, ptr @var
  store volatile float %val2, ptr @var
  store volatile float %val3, ptr @var
  store volatile float %val4, ptr @var
  store volatile float %val5, ptr @var
  store volatile float %val6, ptr @var
  store volatile float %val7, ptr @var
  store volatile float %val8, ptr @var
  store volatile float %val9, ptr @var
  store volatile float %val10, ptr @var
  store volatile float %val11, ptr @var
  store volatile float %val12, ptr @var
  store volatile float %val13, ptr @var
  store volatile float %val14, ptr @var
  store volatile float %val15, ptr @var
  store volatile float %val16, ptr @var
  store volatile float %val17, ptr @var
  store volatile float %val18, ptr @var
  store volatile float %val19, ptr @var
  store volatile float %val20, ptr @var
  store volatile float %val21, ptr @var
  store volatile float %val22, ptr @var
  store volatile float %val23, ptr @var
  store volatile float %val24, ptr @var
  store volatile float %val25, ptr @var
  store volatile float %val26, ptr @var
  store volatile float %val27, ptr @var
  store volatile float %val28, ptr @var
  store volatile float %val29, ptr @var
  store volatile float %val30, ptr @var
  store volatile float %val31, ptr @var
  store volatile float %val32, ptr @var

; CHECK: ldp     d9, d8, [sp
; CHECK: ldp     d11, d10, [sp
; CHECK: ldp     d13, d12, [sp
; CHECK: ldp     d15, d14, [sp
  ret void
}
