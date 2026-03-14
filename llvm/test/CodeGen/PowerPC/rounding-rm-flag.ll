; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     -mcpu=pwr9 -stop-after=early-ifcvt < %s | FileCheck %s

define float @test_XSRDPI(float %f) {
entry:
  %0 = tail call float @llvm.round.f32(float %f)
  ret float %0

; CHECK-LABEL: name:            test_XSRDPI
; CHECK-NOT:  %2:vsfrc = nofpexcept XSRDPI killed %1, implicit $rm
; CHECK:      %2:vsfrc = nofpexcept XSRDPI killed %1
}

define double @test_XSRDPIM(double %d) {
entry:
  %0 = tail call double @llvm.floor.f64(double %d)
  ret double %0

; CHECK-LABEL: name:            test_XSRDPIM
; CHECK-NOT:  %1:vsfrc = nofpexcept XSRDPIM %0, implicit $rm
; CHECK:      %1:vsfrc = nofpexcept XSRDPIM %0
}

declare float @llvm.round.f32(float)
declare double @llvm.floor.f64(double)

