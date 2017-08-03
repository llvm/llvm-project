; RUN: llc < %s -march=x86-64 -mcpu=penryn | FileCheck %s

define <4 x float> @foo(<4 x float>* %p, <4 x float> %x) nounwind {
  %t = load <4 x float>, <4 x float>* %p, align 4
  %z = fmul <4 x float> %t, %x
  ret <4 x float> %z
}

; CHECK-LABEL: foo:
; CHECK: movups
; CHECK: ret

define <2 x double> @bar(<2 x double>* %p, <2 x double> %x) nounwind {
  %t = load <2 x double>, <2 x double>* %p, align 8
  %z = fmul <2 x double> %t, %x
  ret <2 x double> %z
}

; CHECK-LABEL: bar:
; CHECK: movupd
; CHECK: ret
