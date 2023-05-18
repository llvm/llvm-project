; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios3.0.0"

define <4 x double> @PR14337(<4 x double> %a, <4 x double> %b) {
  %foo = fadd <4 x double>  %a, %b
  ret <4 x double> %foo
; CHECK-LABEL: PR14337:
; CHECK: vst1.64
; CHECK: vst1.64
}
