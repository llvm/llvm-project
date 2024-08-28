; RUN: llc -mtriple armv7 -target-abi aapcs -float-abi soft -O0 -o - < %s \
; RUN:   | FileCheck %s -check-prefix CHECK-SOFT -check-prefix CHECK
; RUN: llc -mtriple armv7 -target-abi aapcs -float-abi hard -O0 -o - < %s \
; RUN:   | FileCheck %s -check-prefix CHECK-HARD -check-prefix CHECK

; Tests for passing floating-point regs. Variadic functions will always use
; general-purpose registers. Standard functions will use the floating-point
; registers if there is hardware FP available.

declare i1 @non_variadic(float, float, float, float)
declare i1 @non_variadic_big(float, float, float, float, float, float)
declare i1 @variadic(float, ...)

define void @non_variadic_fp(float %x, float %y) {
; CHECK-LABEL: non_variadic_fp:
; CHECK: b non_variadic
entry:
  %call = tail call i1 (float, float, float, float) @non_variadic(float %y, float %x, float %x, float %y)
  ret void
}

define void @variadic_fp(float %x, float %y) {
; CHECK-LABEL: variadic_fp:
; CHECK: b variadic
entry:
  %call = tail call i1 (float, ...) @variadic(float %y, float %x, float %x, float %y)
  ret void
}

; With soft-float, general-purpose registers are used and there are not enough
; of them to handle the 6 arguments. With hard-float, we have plenty of regs
; (s0-s15) to pass FP arguments.
define void @non_variadic_fp_big(float %x, float %y) {
; CHECK-LABEL: non_variadic_fp_big:
; CHECK-SOFT: bl non_variadic_big
; CHECK-HARD: b non_variadic_big
entry:
  %call = tail call i1 (float, float, float, float, float, float) @non_variadic_big(float %y, float %x, float %x, float %y, float %x, float %y)
  ret void
}

; Variadic functions cannot use FP regs to pass arguments; only GP regs.
define void @variadic_fp_big(float %x, float %y) {
; CHECK-LABEL: variadic_fp_big:
; CHECK: bl variadic
entry:
  %call = tail call i1 (float, ...) @variadic(float %y, float %x, float %x, float %y, float %x, float %y)
  ret void
}
