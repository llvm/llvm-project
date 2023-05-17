; RUN: llc -mtriple=riscv32 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | not FileCheck -check-prefix=CHECK-DIV %s
; RUN: llc -mtriple=riscv64 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | not FileCheck -check-prefix=CHECK-DIV %s
; RUN: llc -mtriple=riscv32 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | not FileCheck -check-prefix=CHECK-REM %s
; RUN: llc -mtriple=riscv64 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | not FileCheck -check-prefix=CHECK-REM %s

; RUN: llc -mtriple=riscv32 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | not FileCheck -check-prefix=CHECK-UDIV %s
; RUN: llc -mtriple=riscv64 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | not FileCheck -check-prefix=CHECK-UDIV %s
; RUN: llc -mtriple=riscv32 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | not FileCheck -check-prefix=CHECK-UREM %s
; RUN: llc -mtriple=riscv64 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | not FileCheck -check-prefix=CHECK-UREM %s

; RUN: llc -mtriple=riscv32 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | FileCheck -check-prefix=CHECK-MUL %s
; RUN: llc -mtriple=riscv64 -mattr=+zmmul -verify-machineinstrs < %s \
; RUN:  | FileCheck -check-prefix=CHECK-MUL %s

; RUN: llc -mtriple=riscv32 -mattr=+m -verify-machineinstrs < %s \
; RUN:  | FileCheck -check-prefixes=CHECK-MUL,CHECK-UDIV,CHECK-DIV,CHECK-UREM,CHECK-REM %s
; RUN: llc -mtriple=riscv64 -mattr=+m -verify-machineinstrs < %s \
; RUN:  | FileCheck -check-prefixes=CHECK-MUL,CHECK-UDIV,CHECK-DIV,CHECK-UREM,CHECK-REM %s

define i32 @foo(i32 %a, i32 %b) {
; CHECK-UDIV: divu{{w?}} {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}
  %1 = udiv i32 %a, %b
; CHECK-DIV: div{{w?}} {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}
  %2 = sdiv i32 %a, %1
; CHECK-MUL: mul{{w?}} {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}
  %3 = mul i32 %b, %2
; CHECK-UREM: remu{{w?}} {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}
  %4 = urem i32 %3, %b
; CHECK-REM: rem{{w?}} {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}, {{[as]}}{{[0-9]}}
  %5 = srem i32 %4, %a
  ret i32 %5
}
