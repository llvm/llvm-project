; RUN: llc -verify-machineinstrs < %s -mcpu=a2q | FileCheck %s
target triple = "powerpc64-bgq-linux"

define <4 x double> @foo(<4 x double>* %p) {
entry:
  %v = load <4 x double>, <4 x double>* %p, align 8
  ret <4 x double> %v
}

; CHECK: @foo
; CHECK-DAG: li [[REG1:[0-9]+]], 31
; CHECK-DAG: qvlfdx [[REG4:[0-9]+]], 0, 3
; CHECK-DAG: qvlfdx [[REG2:[0-9]+]], 3, [[REG1]]
; CHECK-DAG: qvlpcldx [[REG3:[0-9]+]], 0, 3
; CHECK-DAG: qvfperm 1, [[REG4]], [[REG2]], [[REG3]]
; CHECK: blr

define <4 x double> @bar(<4 x double>* %p) {
entry:
  %v = load <4 x double>, <4 x double>* %p, align 32
  ret <4 x double> %v
}

; CHECK: @bar
; CHECK: qvlfdx

