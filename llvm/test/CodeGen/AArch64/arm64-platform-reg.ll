; RUN: llc -mtriple=arm64-apple-ios -mattr=+reserve-x18 -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18
; RUN: llc -mtriple=arm64-freebsd-gnu -mattr=+reserve-x18 -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18
; RUN: llc -mtriple=aarch64-fuchsia -mattr=+reserve-x20 -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X20
; RUN: llc -mtriple=aarch64-fuchsia -mattr=+reserve-x18,+reserve-x20 -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18 --check-prefix=CHECK-RESERVE-X20
; RUN: llc -mtriple=arm64-linux-gnu -o - %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-android -o - %s | FileCheck %s --check-prefix=CHECK-RESERVE --check-prefix=CHECK-RESERVE-X18
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
; RUN: -mattr=+reserve-x18 \
; RUN: -mattr=+reserve-x20 \
; RUN: -o - %s | FileCheck %s \
; RUN: --check-prefix=CHECK-RESERVE \
; RUN: --check-prefix=CHECK-RESERVE-X1 \
; RUN: --check-prefix=CHECK-RESERVE-X2 \
; RUN: --check-prefix=CHECK-RESERVE-X3 \
; RUN: --check-prefix=CHECK-RESERVE-X4 \
; RUN: --check-prefix=CHECK-RESERVE-X5 \
; RUN: --check-prefix=CHECK-RESERVE-X6 \
; RUN: --check-prefix=CHECK-RESERVE-X7 \
; RUN: --check-prefix=CHECK-RESERVE-X18 \
; RUN: --check-prefix=CHECK-RESERVE-X20

; x18 is reserved as a platform register on Darwin but not on other
; systems. Create loads of register pressure and make sure this is respected.

; Also, fp must always refer to a valid frame record, even if it's not the one
; of the current function, so it shouldn't be used either.

@var = global [30 x i64] zeroinitializer

define void @keep_live() {
  %val = load volatile [30 x i64], [30 x i64]* @var
  store volatile [30 x i64] %val, [30 x i64]* @var

; CHECK: ldr x18
; CHECK: str x18

; CHECK-RESERVE-NOT: ldr fp
; CHECK-RESERVE-X1-NOT: ldr x1,
; CHECK-RESERVE-X2-NOT: ldr x2,
; CHECK-RESERVE-X3-NOT: ldr x3,
; CHECK-RESERVE-X4-NOT: ldr x4,
; CHECK-RESERVE-X5-NOT: ldr x5,
; CHECK-RESERVE-X6-NOT: ldr x6,
; CHECK-RESERVE-X7-NOT: ldr x7,
; CHECK-RESERVE-X18-NOT: ldr x18
; CHECK-RESERVE-X20-NOT: ldr x20
; CHECK-RESERVE: Spill
; CHECK-RESERVE-NOT: ldr fp
; CHECK-RESERVE-X1-NOT: ldr x1,
; CHECK-RESERVE-X2-NOT: ldr x2,
; CHECK-RESERVE-X3-NOT: ldr x3,
; CHECK-RESERVE-X4-NOT: ldr x4,
; CHECK-RESERVE-X5-NOT: ldr x5,
; CHECK-RESERVE-X6-NOT: ldr x6,
; CHECK-RESERVE-X7-NOT: ldr x7,
; CHECK-RESERVE-X18-NOT: ldr x18
; CHECK-RESERVE-X20-NOT: ldr x20
; CHECK-RESERVE: ret
  ret void
}
