; RUN: llc -mtriple=arm64-eabi -o - %s | FileCheck %s

define { i192, i192, i21, i192 } @foo(i192) {
; CHECK-LABEL: foo:
; CHECK: stp xzr, xzr, [x8]
  ret { i192, i192, i21, i192 } {i192 0, i192 1, i21 2, i192 3}
}
