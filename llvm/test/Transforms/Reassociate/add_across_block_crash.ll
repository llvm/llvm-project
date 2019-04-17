; RUN: opt < %s -reassociate -S | FileCheck %s

; This test is to make sure while processing a block, uses of instructions
; from a different basic block don't get added to be re-optimized

define  void @main() {
; CHECK-LABEL: @main(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 undef, label %bb1, label %bb2
; CHECK:       bb1:
; CHECK-NEXT:    ret void
; CHECK:       bb2:
; CHECK-NEXT:    ret void
;
entry:
  %0 = fadd fast float undef, undef
  br i1 undef, label %bb1, label %bb2

bb1:
  %1 = fmul fast float undef, -2.000000e+00
  %2 = fmul fast float %1, 2.000000e+00
  %3 = fadd fast float %2, 2.000000e+00
  %4 = fadd fast float %3, %0
  %mul351 = fmul fast float %4, 5.000000e-01
  ret void

bb2:
  ret void
}

