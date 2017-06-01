; RUN: llc -verify-machineinstrs < %s -march=ppc64 -mcpu=a2q | FileCheck %s
target triple = "powerpc64-bgq-linux"

define void @foo(<4 x double> %v, <4 x double>* %p) {
entry:
  store <4 x double> %v, <4 x double>* %p, align 8
  ret void
}

; CHECK: @foo
; CHECK: stfd
; CHECK: stfd
; CHECK: stfd
; CHECK: stfd
; CHECK: blr

define void @bar(<4 x double> %v, <4 x double>* %p) {
entry:
  store <4 x double> %v, <4 x double>* %p, align 32
  ret void
}

; CHECK: @bar
; CHECK: qvstfdx

