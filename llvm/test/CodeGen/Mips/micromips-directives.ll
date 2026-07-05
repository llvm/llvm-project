; This test checks if the '.set [no]micromips' directives
; are emitted before a function's entry label.

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+micromips %s -o - | \
; RUN:   FileCheck %s -check-prefix=CHECK-MM
; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=-micromips %s -o - | \
; RUN:   FileCheck %s -check-prefix=CHECK-NO-MM

define i32 @main() nounwind {
entry:
  ret i32 0
}

; CHECK-MM: .set micromips
; CHECK-NO-MM: .set nomicromips
; CHECK: main:
