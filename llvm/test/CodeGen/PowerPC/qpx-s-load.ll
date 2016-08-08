; RUN: llc -verify-machineinstrs < %s -march=ppc64 -mcpu=a2q | FileCheck %s
target triple = "powerpc64-bgq-linux"

define <4 x float> @foo(<4 x float>* %p) {
entry:
  %v = load <4 x float>, <4 x float>* %p, align 4
  ret <4 x float> %v
}

; CHECK: @foo
; CHECK-DAG: li [[REG1:[0-9]+]], 15
; CHECK-DAG: qvlfsx [[REG4:[0-9]+]], 0, 3
; CHECK-DAG: qvlfsx [[REG2:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: qvlpclsx [[REG3:[0-9]+]], 0, 3
; CHECK-DAG: qvfperm 1, [[REG4]], [[REG2]], [[REG3]]
; CHECK: blr

define <4 x float> @bar(<4 x float>* %p) {
entry:
  %v = load <4 x float>, <4 x float>* %p, align 16
  ret <4 x float> %v
}

; CHECK: @bar
; CHECK: qvlfsx

