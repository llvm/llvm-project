; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x18 -mattr=+call-saved-x18 \
; RUN: -o - %s | FileCheck %s

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x18 -mattr=+call-saved-x18 \
; RUN: -global-isel \
; RUN: -o - %s | FileCheck %s

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x18 -mattr=+call-saved-x18 \
; RUN: -fast-isel \
; RUN: -o - %s | FileCheck %s

; Used to exhaust the supply of GPRs.
@var = global [30 x i64] zeroinitializer

; If a register is specified to be both reserved and callee-saved, then it
; should not be allocated and should not be spilled onto the stack.
define void @foo() {
; CHECK-NOT: str x18, [sp

  %val = load volatile [30 x i64], [30 x i64]* @var
  store volatile [30 x i64] %val, [30 x i64]* @var

; CHECK-NOT: ldr x18
; CHECK-NOT: str x18

; CHECK-NOT: ldr x18, [sp
  ret void
}
