; RUN: llc -verify-machineinstrs < %s -mcpu=a2q | FileCheck %s
target triple = "powerpc64-bgq-linux"

define void @foo(<4 x float> %v, <4 x float>* %p) {
entry:
  store <4 x float> %v, <4 x float>* %p, align 4
  ret void
}

; CHECK: @foo
; CHECK: stfs
; CHECK: stfs
; CHECK: stfs
; CHECK: stfs
; CHECK: blr

define void @bar(<4 x float> %v, <4 x float>* %p) {
entry:
  store <4 x float> %v, <4 x float>* %p, align 16
  ret void
}

; CHECK: @bar
; CHECK: qvstfsx

