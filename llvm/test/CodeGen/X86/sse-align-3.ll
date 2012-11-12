; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; CHECK-NOT:     movapd
; CHECK:     movaps
; CHECK-NOT:     movapd
; CHECK:     movaps
; CHECK-NOT:     movap

define void @foo(<4 x float>* %p, <4 x float> %x) nounwind {
  store <4 x float> %x, <4 x float>* %p
  ret void
}
define void @bar(<2 x double>* %p, <2 x double> %x) nounwind {
  store <2 x double> %x, <2 x double>* %p
  ret void
}
