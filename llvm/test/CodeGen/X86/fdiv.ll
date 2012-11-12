; RUN: llc < %s -march=x86-64 -enable-unsafe-fp-math | FileCheck %s

define double @exact(double %x) {
; Exact division by a constant converted to multiplication.
; CHECK: @exact
; CHECK: mulsd
  %div = fdiv double %x, 2.0
  ret double %div
}

define double @inexact(double %x) {
; Inexact division by a constant converted to multiplication.
; CHECK: @inexact
; CHECK: mulsd
  %div = fdiv double %x, 0x41DFFFFFFFC00000 
  ret double %div
}

define double @funky(double %x) {
; No conversion to multiplication if too funky.
; CHECK: @funky
; CHECK: divsd
  %div = fdiv double %x, 0.0
  ret double %div
}

define double @denormal1(double %x) {
; Don't generate multiplication by a denormal.
; CHECK: @denormal1
; CHECK: divsd
  %div = fdiv double %x, 0x7FD0000000000001
  ret double %div
}

define double @denormal2(double %x) {
; Don't generate multiplication by a denormal.
; CHECK: @denormal
; CHECK: divsd
  %div = fdiv double %x, 0x7FEFFFFFFFFFFFFF
  ret double %div
}
