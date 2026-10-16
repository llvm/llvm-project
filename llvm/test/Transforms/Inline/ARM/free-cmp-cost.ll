; RUN: opt -mtriple=arm-unknown-linux-gnu -passes="print<inline-cost>" -disable-output %s 2>&1 | FileCheck %s

; CHECK:      Analyzing call of inner_icmp... (caller:outer_icmp)
; CHECK:      NumInstructionsSimplified: 2
; CHECK:      Cost: -35
; CHECK:      Analyzing call of inner_fcmp... (caller:outer_fcmp)
; CHECK:      NumInstructionsSimplified: 2
; CHECK:      Cost: -35
; CHECK:      Analyzing call of inner_icmp_not_minmax... (caller:outer_icmp_not_minmax)
; CHECK:      NumInstructionsSimplified: 1
; CHECK:      Cost: -35
; CHECK:      Analyzing call of inner_fcmp_not_minmax... (caller:outer_fcmp_not_minmax)
; CHECK:      NumInstructionsSimplified: 1
; CHECK:      Cost: -35

define <4 x i32> @outer_icmp(<4 x i32> %a, <4 x i32> %b) {
  %r = call <4 x i32> @inner_icmp(<4 x i32> %a, <4 x i32> %b)
  ret <4 x i32> %r
}

define <4 x float> @outer_fcmp(<4 x float> %a, <4 x float> %b) {
  %r = call <4 x float> @inner_fcmp(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %r
}

define <4 x i32> @outer_icmp_not_minmax(<4 x i32> %a, <4 x i32> %b, <4 x i32> %x) {
  %r = call <4 x i32> @inner_icmp_not_minmax(<4 x i32> %a, <4 x i32> %b, <4 x i32> %x)
  ret <4 x i32> %r
}

define <4 x float> @outer_fcmp_not_minmax(<4 x float> %a, <4 x float> %b, <4 x float> %x) {
  %r = call <4 x float> @inner_fcmp_not_minmax(<4 x float> %a, <4 x float> %b, <4 x float> %x)
  ret <4 x float> %r
}

define <4 x i32> @inner_icmp(<4 x i32> %a, <4 x i32> %b) {
  %c = icmp slt <4 x i32> %a, %b
  %s = select <4 x i1> %c, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %s
}

define <4 x float> @inner_fcmp(<4 x float> %a, <4 x float> %b) {
  %c = fcmp olt <4 x float> %a, %b
  %s = select nnan nsz <4 x i1> %c, <4 x float> %a, <4 x float> %b
  ret <4 x float> %s
}

; Negative test: the select is not a min/max, so the compare is not free.
define <4 x i32> @inner_icmp_not_minmax(<4 x i32> %a, <4 x i32> %b, <4 x i32> %x) {
  %c = icmp slt <4 x i32> %a, %b
  %s = select <4 x i1> %c, <4 x i32> %a, <4 x i32> %x
  ret <4 x i32> %s
}

; Negative test: the select is not a min/max, so the compare is not free.
define <4 x float> @inner_fcmp_not_minmax(<4 x float> %a, <4 x float> %b, <4 x float> %x) {
  %c = fcmp olt <4 x float> %a, %b
  %s = select nnan nsz <4 x i1> %c, <4 x float> %a, <4 x float> %x
  ret <4 x float> %s
}
