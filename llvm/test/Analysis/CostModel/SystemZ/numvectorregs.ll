; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s

; Test that the cost for the number of vector registers is returned for a
; non-power-of-two vector type.
define <6 x double> @fun0(<6 x double> %lhs, <6 x double> %rhs) {
  %a = fadd <6 x double> %lhs, %rhs
  ret <6 x double> %a
; CHECK: Printing analysis 'Cost Model Analysis' for function 'fun0':
; CHECK: Cost Model: Found an estimated cost of 3 for instruction:   %a = fadd <6 x double>
}
