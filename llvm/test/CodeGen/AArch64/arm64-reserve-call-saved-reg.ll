; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x9 -mattr=+call-saved-x9 -o - %s | FileCheck %s --check-prefixes=CHECK-X9
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x9 -mattr=+call-saved-x9 -global-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X9
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x9 -mattr=+call-saved-x9 -fast-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X9

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x10 -mattr=+call-saved-x10 -o - %s | FileCheck %s --check-prefixes=CHECK-X10
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x10 -mattr=+call-saved-x10 -global-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X10
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x10 -mattr=+call-saved-x10 -fast-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X10

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x11 -mattr=+call-saved-x11 -o - %s | FileCheck %s --check-prefixes=CHECK-X11
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x11 -mattr=+call-saved-x11 -global-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X11
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x11 -mattr=+call-saved-x11 -fast-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X11

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x12 -mattr=+call-saved-x12 -o - %s | FileCheck %s --check-prefixes=CHECK-X12
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x12 -mattr=+call-saved-x12 -global-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X12
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x12 -mattr=+call-saved-x12 -fast-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X12

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x13 -mattr=+call-saved-x13 -o - %s | FileCheck %s --check-prefixes=CHECK-X13
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x13 -mattr=+call-saved-x13 -global-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X13
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x13 -mattr=+call-saved-x13 -fast-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X13

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x14 -mattr=+call-saved-x14 -o - %s | FileCheck %s --check-prefixes=CHECK-X14
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x14 -mattr=+call-saved-x14 -global-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X14
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x14 -mattr=+call-saved-x14 -fast-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X14

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x15 -mattr=+call-saved-x15 -o - %s | FileCheck %s --check-prefixes=CHECK-X15
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x15 -mattr=+call-saved-x15 -global-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X15
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x15 -mattr=+call-saved-x15 -fast-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X15

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x18 -mattr=+call-saved-x18 -o - %s | FileCheck %s --check-prefixes=CHECK-X18
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x18 -mattr=+call-saved-x18 -global-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X18
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x18 -mattr=+call-saved-x18 -fast-isel -o - %s | FileCheck %s --check-prefixes=CHECK-X18

; Used to exhaust the supply of GPRs.
@var = global [30 x i64] zeroinitializer

; If a register is specified to be both reserved and callee-saved, then it
; should not be allocated and should not be spilled onto the stack.
define void @foo() {
; CHECK-X9-NOT: str x9, [sp
; CHECK-X10-NOT: str x10, [sp
; CHECK-X11-NOT: str x11, [sp
; CHECK-X12-NOT: str x12, [sp
; CHECK-X13-NOT: str x13, [sp
; CHECK-X14-NOT: str x14, [sp
; CHECK-X15-NOT: str x15, [sp
; CHECK-X18-NOT: str x18, [sp

  %val = load volatile [30 x i64], ptr @var
  store volatile [30 x i64] %val, ptr @var

; CHECK-X9-NOT: ldr x9
; CHECK-X10-NOT: ldr x10
; CHECK-X11-NOT: ldr x11
; CHECK-X12-NOT: ldr x12
; CHECK-X13-NOT: ldr x13
; CHECK-X14-NOT: ldr x14
; CHECK-X15-NOT: ldr x15
; CHECK-X18-NOT: ldr x18
; CHECK-X9-NOT: str x9
; CHECK-X10-NOT: str x10
; CHECK-X11-NOT: str x11
; CHECK-X12-NOT: str x12
; CHECK-X13-NOT: str x13
; CHECK-X14-NOT: str x14
; CHECK-X15-NOT: str x15
; CHECK-X18-NOT: str x18

; CHECK-X9-NOT: ldr x9, [sp
; CHECK-X10-NOT: ldr x10, [sp
; CHECK-X11-NOT: ldr x11, [sp
; CHECK-X12-NOT: ldr x12, [sp
; CHECK-X13-NOT: ldr x13, [sp
; CHECK-X14-NOT: ldr x14, [sp
; CHECK-X15-NOT: ldr x15, [sp
; CHECK-X18-NOT: ldr x18, [sp
  ret void
}
