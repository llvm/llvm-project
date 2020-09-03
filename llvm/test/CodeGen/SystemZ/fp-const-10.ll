; Test loads of SNaN.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test that we don't do an FP extending load, as this would result in a
; converstion to QNaN.
define double @f1() {
; CHECK-LABEL: .LCPI0_0
; CHECK:      .quad   0x7ff4000000000000
; CHECK-LABEL: f1:
; CHECK:      larl    %r1, .LCPI0_0
; CHECK-NOT:  ldeb    %f0, 0(%r1)
; CHECK:      ld      %f0, 0(%r1)
  ret double 0x7FF4000000000000
}
