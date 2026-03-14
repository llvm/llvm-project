; RUN: llc < %s -o - | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-os"

declare <4 x double> @user_func(<4 x double>) #1

; Make sure we are not crashing on this code.
; CHECK-LABEL: caller_function
; CHECK: ret
define void @caller_function(<4 x double>, <4 x double>, <4 x double>, <4 x double>, <4 x double>) #1 {
entry:
  %r = call <4 x double> @user_func(<4 x double> %4)
  ret void
}

attributes #1 = { nounwind readnone }

