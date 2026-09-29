; RUN: llc -mtriple=arm64-linux-gnu -o - %s | FileCheck %s
; RUN: llc -mtriple=arm64-linux-gnu -O0 -fast-isel -o - %s | FileCheck %s

; O0 checked for fastisel purposes. It has a separate path which
; creates a constpool entry for floating values.

define double @needs_const() {
  ret double 3.14159
; CHECK: .LCPI0_0:

; CHECK: adrp {{x[0-9]+}}, .LCPI0_0
; CHECK: ldr d0, [{{x[0-9]+}}, :lo12:.LCPI0_0]
}
