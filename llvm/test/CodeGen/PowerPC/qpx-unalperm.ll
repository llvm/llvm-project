; RUN: llc -verify-machineinstrs < %s -mcpu=a2q | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-bgq-linux"

define <4 x double> @foo(<4 x double>* %a) {
entry:
  %r = load <4 x double>, <4 x double>* %a, align 32
  ret <4 x double> %r
; CHECK: qvlfdx
; CHECK: blr
}

define <4 x double> @bar(<4 x double>* %a) {
entry:
  %r = load <4 x double>, <4 x double>* %a, align 8
  %b = getelementptr <4 x double>, <4 x double>* %a, i32 16
  %s = load <4 x double>, <4 x double>* %b, align 32
  %t = fadd <4 x double> %r, %s
  ret <4 x double> %t
; CHECK: qvlpcldx
; CHECK: qvlfdx
; CHECK: qvfperm
; CHECK: blr
}

define <4 x double> @bar1(<4 x double>* %a) {
entry:
  %r = load <4 x double>, <4 x double>* %a, align 8
  %b = getelementptr <4 x double>, <4 x double>* %a, i32 16
  %s = load <4 x double>, <4 x double>* %b, align 8
  %t = fadd <4 x double> %r, %s
  ret <4 x double> %t
}

define <4 x double> @bar2(<4 x double>* %a) {
entry:
  %r = load <4 x double>, <4 x double>* %a, align 8
  %b = getelementptr <4 x double>, <4 x double>* %a, i32 1
  %s = load <4 x double>, <4 x double>* %b, align 32
  %t = fadd <4 x double> %r, %s
  ret <4 x double> %t
}

define <4 x double> @bar3(<4 x double>* %a) {
entry:
  %r = load <4 x double>, <4 x double>* %a, align 8
  %b = getelementptr <4 x double>, <4 x double>* %a, i32 1
  %s = load <4 x double>, <4 x double>* %b, align 8
  %t = fadd <4 x double> %r, %s
  ret <4 x double> %t
}

define <4 x double> @bar4(<4 x double>* %a) {
entry:
  %r = load <4 x double>, <4 x double>* %a, align 8
  %b = getelementptr <4 x double>, <4 x double>* %a, i32 1
  %s = load <4 x double>, <4 x double>* %b, align 8
  %c = getelementptr <4 x double>, <4 x double>* %b, i32 1
  %t = load <4 x double>, <4 x double>* %c, align 8
  %u = fadd <4 x double> %r, %s
  %v = fadd <4 x double> %u, %t
  ret <4 x double> %v
}

