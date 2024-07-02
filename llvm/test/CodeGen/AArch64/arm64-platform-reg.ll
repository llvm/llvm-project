; RUN: llc -mtriple=arm64-apple-ios -mattr=+reserve-x18 -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18
; RUN: llc -mtriple=arm64-freebsd-gnu -mattr=+reserve-x18 -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18
; RUN: llc -mtriple=arm64-linux-gnu -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-android -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18
; RUN: llc -mtriple=aarch64-linux-ohos -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18
; RUN: llc -mtriple=aarch64-fuchsia -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18
; RUN: llc -mtriple=aarch64-windows -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18

; Test reserve-x# options individually.
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x1 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X1
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x2 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X2
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x3 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X3
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x4 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X4
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x5 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X5
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x6 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X6
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x7 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X7
; RUN: llc -mtriple=arm64-linux-gnu -reserve-regs-for-regalloc=X8 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X8
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x9 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X9
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x10 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X10
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x11 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X11
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x12 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X12
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x13 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X13
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x14 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X14
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x15 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X15
; RUN: llc -mtriple=arm64-linux-gnu -reserve-regs-for-regalloc=X16 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X16
; RUN: llc -mtriple=arm64-linux-gnu -reserve-regs-for-regalloc=X17 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X17
; RUN: llc -mtriple=arm64-linux-gnu -reserve-regs-for-regalloc=X19 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X19
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x20 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X20
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x21 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X21
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x22 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X22
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x23 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X23
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x24 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X24
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x25 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X25
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x26 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X26
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x27 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X27
; RUN: llc -mtriple=arm64-linux-gnu -mattr=+reserve-x28 -o - %s | FileCheck %s --check-prefixes=CHECK-RESERVE,CHECK-RESERVE-X28

; Test multiple of reserve-x# options together.
; RUN: llc -mtriple=arm64-linux-gnu \
; RUN: -mattr=+reserve-x1 \
; RUN: -mattr=+reserve-x2 \
; RUN: -mattr=+reserve-x18 \
; RUN: -o - %s | FileCheck %s \
; RUN: --check-prefix=CHECK-RESERVE \
; RUN: --check-prefix=CHECK-RESERVE-X1 \
; RUN: --check-prefix=CHECK-RESERVE-X2 \
; RUN: --check-prefix=CHECK-RESERVE-X18

; Test all reserve-x# options together.
; RUN: llc -mtriple=arm64-linux-gnu \
; RUN: -mattr=+reserve-x1 \
; RUN: -mattr=+reserve-x2 \
; RUN: -mattr=+reserve-x3 \
; RUN: -mattr=+reserve-x4 \
; RUN: -mattr=+reserve-x5 \
; RUN: -mattr=+reserve-x6 \
; RUN: -mattr=+reserve-x7 \
; RUN: -mattr=+reserve-x9 \
; RUN: -mattr=+reserve-x10 \
; RUN: -mattr=+reserve-x11 \
; RUN: -mattr=+reserve-x12 \
; RUN: -mattr=+reserve-x13 \
; RUN: -mattr=+reserve-x14 \
; RUN: -mattr=+reserve-x15 \
; RUN: -mattr=+reserve-x18 \
; RUN: -mattr=+reserve-x20 \
; RUN: -mattr=+reserve-x21 \
; RUN: -mattr=+reserve-x22 \
; RUN: -mattr=+reserve-x23 \
; RUN: -mattr=+reserve-x24 \
; RUN: -mattr=+reserve-x25 \
; RUN: -mattr=+reserve-x26 \
; RUN: -mattr=+reserve-x27 \
; RUN: -mattr=+reserve-x28 \
; RUN: -reserve-regs-for-regalloc=X8,X16,X17,X19 \
; RUN: -o - %s | FileCheck %s \
; RUN: --check-prefix=CHECK-RESERVE \
; RUN: --check-prefix=CHECK-RESERVE-X1 \
; RUN: --check-prefix=CHECK-RESERVE-X2 \
; RUN: --check-prefix=CHECK-RESERVE-X3 \
; RUN: --check-prefix=CHECK-RESERVE-X4 \
; RUN: --check-prefix=CHECK-RESERVE-X5 \
; RUN: --check-prefix=CHECK-RESERVE-X6 \
; RUN: --check-prefix=CHECK-RESERVE-X7 \
; RUN: --check-prefix=CHECK-RESERVE-X8 \
; RUN: --check-prefix=CHECK-RESERVE-X9 \
; RUN: --check-prefix=CHECK-RESERVE-X10 \
; RUN: --check-prefix=CHECK-RESERVE-X11 \
; RUN: --check-prefix=CHECK-RESERVE-X12 \
; RUN: --check-prefix=CHECK-RESERVE-X13 \
; RUN: --check-prefix=CHECK-RESERVE-X14 \
; RUN: --check-prefix=CHECK-RESERVE-X15 \
; RUN: --check-prefix=CHECK-RESERVE-X16 \
; RUN: --check-prefix=CHECK-RESERVE-X17 \
; RUN: --check-prefix=CHECK-RESERVE-X18 \
; RUN: --check-prefix=CHECK-RESERVE-X19 \
; RUN: --check-prefix=CHECK-RESERVE-X20 \
; RUN: --check-prefix=CHECK-RESERVE-X21 \
; RUN: --check-prefix=CHECK-RESERVE-X22 \
; RUN: --check-prefix=CHECK-RESERVE-X23 \
; RUN: --check-prefix=CHECK-RESERVE-X24 \
; RUN: --check-prefix=CHECK-RESERVE-X25 \
; RUN: --check-prefix=CHECK-RESERVE-X26 \
; RUN: --check-prefix=CHECK-RESERVE-X27 \
; RUN: --check-prefix=CHECK-RESERVE-X28

; x18 is reserved as a platform register on Darwin but not on other
; systems. Create loads of register pressure and make sure this is respected.

; Also, fp must always refer to a valid frame record, even if it's not the one
; of the current function, so it shouldn't be used either.

@var = global [30 x i64] zeroinitializer

define void @keep_live() {
  %val = load volatile [30 x i64], ptr @var
  store volatile [30 x i64] %val, ptr @var

; CHECK: ldr x18
; CHECK: str x18

; CHECK-RESERVE-NOT: ldr fp
; CHECK-RESERVE-X8-NOT: adrp x8
; CHECK-RESERVE-X8-NOT: ldr x8
; CHECK-RESERVE-X1-NOT: ldr x1,
; CHECK-RESERVE-X2-NOT: ldr x2,
; CHECK-RESERVE-X3-NOT: ldr x3,
; CHECK-RESERVE-X4-NOT: ldr x4,
; CHECK-RESERVE-X5-NOT: ldr x5,
; CHECK-RESERVE-X6-NOT: ldr x6,
; CHECK-RESERVE-X7-NOT: ldr x7,
; CHECK-RESERVE-X9-NOT: ldr x9,
; CHECK-RESERVE-X10-NOT: ldr x10,
; CHECK-RESERVE-X11-NOT: ldr x11,
; CHECK-RESERVE-X12-NOT: ldr x12,
; CHECK-RESERVE-X13-NOT: ldr x13,
; CHECK-RESERVE-X14-NOT: ldr x14,
; CHECK-RESERVE-X15-NOT: ldr x15,
; CHECK-RESERVE-X16-NOT: ldr x16
; CHECK-RESERVE-X17-NOT: ldr x17
; CHECK-RESERVE-X18-NOT: ldr x18
; CHECK-RESERVE-X19-NOT: ldr x19
; CHECK-RESERVE-X20-NOT: ldr x20
; CHECK-RESERVE-X21-NOT: ldr x21
; CHECK-RESERVE-X22-NOT: ldr x22
; CHECK-RESERVE-X23-NOT: ldr x23
; CHECK-RESERVE-X24-NOT: ldr x24
; CHECK-RESERVE-X25-NOT: ldr x25
; CHECK-RESERVE-X26-NOT: ldr x26
; CHECK-RESERVE-X27-NOT: ldr x27
; CHECK-RESERVE-X28-NOT: ldr x28
; CHECK-RESERVE: Spill
; CHECK-RESERVE-NOT: ldr fp
; CHECK-RESERVE-X1-NOT: ldr x1,
; CHECK-RESERVE-X2-NOT: ldr x2,
; CHECK-RESERVE-X3-NOT: ldr x3,
; CHECK-RESERVE-X4-NOT: ldr x4,
; CHECK-RESERVE-X5-NOT: ldr x5,
; CHECK-RESERVE-X6-NOT: ldr x6,
; CHECK-RESERVE-X7-NOT: ldr x7,
; CHECK-RESERVE-X9-NOT: ldr x9,
; CHECK-RESERVE-X10-NOT: ldr x10,
; CHECK-RESERVE-X11-NOT: ldr x11,
; CHECK-RESERVE-X12-NOT: ldr x12,
; CHECK-RESERVE-X13-NOT: ldr x13,
; CHECK-RESERVE-X14-NOT: ldr x14,
; CHECK-RESERVE-X15-NOT: ldr x15,
; CHECK-RESERVE-X16-NOT: ldr x16
; CHECK-RESERVE-X17-NOT: ldr x17
; CHECK-RESERVE-X18-NOT: ldr x18
; CHECK-RESERVE-X19-NOT: ldr x19
; CHECK-RESERVE-X20-NOT: ldr x20
; CHECK-RESERVE-X21-NOT: ldr x21
; CHECK-RESERVE-X22-NOT: ldr x22
; CHECK-RESERVE-X23-NOT: ldr x23
; CHECK-RESERVE-X24-NOT: ldr x24
; CHECK-RESERVE-X25-NOT: ldr x25
; CHECK-RESERVE-X26-NOT: ldr x26
; CHECK-RESERVE-X27-NOT: ldr x27
; CHECK-RESERVE-X28-NOT: ldr x28
; CHECK-RESERVE: ret
  ret void
}
