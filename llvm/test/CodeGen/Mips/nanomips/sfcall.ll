; RUN: llc -mtriple=nanomips --mattr +soft-float -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

; Make sure that soft-float calls are done with BALC_NM instruction.
define double @soft_float_call(double %a, double %b) {
; CHECK: balc __adddf3
; CHECK: BALC_NM
  %add = fadd double %a, %b
  ret double %add
}
