; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
;
; Check that a multiply-and-add results.

define void @f1(float %arg, ptr %Dst) {
; CHECK-LABEL: f1:
; CHECK: maeb
bb:
  %i = fmul contract float %arg, 0xBE6777A5C0000000
  %i4 = fadd contract float %i, 1.000000e+00
  %i5 = fmul contract float %arg, 0xBE6777A5C0000000
  %i6 = fadd contract float %i5, 1.000000e+00
  %i7 = fmul contract float %i4, 2.000000e+00
  store float %i7, ptr %Dst
  ret void
}

