; RUN: llc -march=lanai < %s | FileCheck %s

; Tests that lowering wide registers (128 bits or more) works on Lanai.
; The emitted assembly is not checked, we just do a smoketest.

target datalayout = "E-m:e-p:32:32-i64:64-a:0:32-n32-S64"
target triple = "lanai"

; CHECK-LABEL: add128:
define i128 @add128(i128 %x, i128 %y) {
  %a = add i128 %x, %y
  ret i128 %a
}
