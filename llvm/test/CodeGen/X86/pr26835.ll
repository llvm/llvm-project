; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux"

; CHECK-LABEL: foo
; CHECK: div
define i24 @foo(i24 %a, i24 %b) {
  %r = urem i24 %a, %b
  ret i24 %r
}
