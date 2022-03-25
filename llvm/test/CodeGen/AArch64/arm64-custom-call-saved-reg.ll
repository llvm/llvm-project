; RUN: llc -mtriple=arm64-linux-gnu -mattr=+call-saved-x8 -o - %s \
; RUN: | FileCheck %s --check-prefix=CHECK-SAVED-X8

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+call-saved-x9 -o - %s \
; RUN: | FileCheck %s --check-prefix=CHECK-SAVED-X9

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+call-saved-x10 -o - %s \
; RUN: | FileCheck %s --check-prefix=CHECK-SAVED-X10

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+call-saved-x11 -o - %s \
; RUN: | FileCheck %s --check-prefix=CHECK-SAVED-X11

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+call-saved-x12 -o - %s \
; RUN: | FileCheck %s --check-prefix=CHECK-SAVED-X12

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+call-saved-x13 -o - %s \
; RUN: | FileCheck %s --check-prefix=CHECK-SAVED-X13

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+call-saved-x14 -o - %s \
; RUN: | FileCheck %s --check-prefix=CHECK-SAVED-X14

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+call-saved-x15 -o - %s \
; RUN: | FileCheck %s --check-prefix=CHECK-SAVED-X15

; RUN: llc -mtriple=arm64-linux-gnu -mattr=+call-saved-x18 -o - %s \
; RUN: | FileCheck %s --check-prefix=CHECK-SAVED-X18

; Test all call-saved-x# options together.
; RUN: llc -mtriple=arm64-linux-gnu \
; RUN: -mattr=+call-saved-x8 \
; RUN: -mattr=+call-saved-x9 \
; RUN: -mattr=+call-saved-x10 \
; RUN: -mattr=+call-saved-x11 \
; RUN: -mattr=+call-saved-x12 \
; RUN: -mattr=+call-saved-x13 \
; RUN: -mattr=+call-saved-x14 \
; RUN: -mattr=+call-saved-x15 \
; RUN: -mattr=+call-saved-x18 \
; RUN: -o - %s | FileCheck %s \
; RUN: --check-prefix=CHECK-SAVED-ALL

; Test GlobalISel.
; RUN: llc -mtriple=arm64-linux-gnu \
; RUN: -mattr=+call-saved-x8 \
; RUN: -mattr=+call-saved-x9 \
; RUN: -mattr=+call-saved-x10 \
; RUN: -mattr=+call-saved-x11 \
; RUN: -mattr=+call-saved-x12 \
; RUN: -mattr=+call-saved-x13 \
; RUN: -mattr=+call-saved-x14 \
; RUN: -mattr=+call-saved-x15 \
; RUN: -mattr=+call-saved-x18 \
; RUN: -global-isel \
; RUN: -o - %s | FileCheck %s \
; RUN: --check-prefix=CHECK-SAVED-ALL-GISEL

; Used to exhaust the supply of GPRs.
@var = dso_local global [30 x i64] zeroinitializer

; Check that callee preserves additional CSRs.
define dso_local void @callee() {
; CHECK-LABEL: callee

; CHECK-SAVED-X8: str x8, [sp
; CHECK-SAVED-X9: str x9, [sp
; CHECK-SAVED-X10: str x10, [sp
; CHECK-SAVED-X11: str x11, [sp
; CHECK-SAVED-X12: str x12, [sp
; CHECK-SAVED-X13: str x13, [sp
; CHECK-SAVED-X14: str x14, [sp
; CHECK-SAVED-X15: str x15, [sp
; CHECK-SAVED-X18: str x18, [sp

; CHECK-SAVED-ALL: str x18, [sp
; CHECK-SAVED-ALL-NEXT: .cfi_def_cfa_offset
; CHECK-SAVED-ALL-NEXT: stp x15, x14, [sp
; CHECK-SAVED-ALL-NEXT: stp x13, x12, [sp
; CHECK-SAVED-ALL-NEXT: stp x11, x10, [sp
; CHECK-SAVED-ALL-NEXT: stp x9, x8, [sp

  %val = load volatile [30 x i64], [30 x i64]* @var
  store volatile [30 x i64] %val, [30 x i64]* @var

; CHECK-SAVED-ALL: ldp x9, x8, [sp
; CHECK-SAVED-ALL-NEXT: ldp x11, x10, [sp
; CHECK-SAVED-ALL-NEXT: ldp x13, x12, [sp
; CHECK-SAVED-ALL-NEXT: ldp x15, x14, [sp
; CHECK-SAVED-ALL-NEXT: ldr x18, [sp

; CHECK-SAVED-X8: ldr x8, [sp
; CHECK-SAVED-X9: ldr x9, [sp
; CHECK-SAVED-X10: ldr x10, [sp
; CHECK-SAVED-X11: ldr x11, [sp
; CHECK-SAVED-X12: ldr x12, [sp
; CHECK-SAVED-X13: ldr x13, [sp
; CHECK-SAVED-X14: ldr x14, [sp
; CHECK-SAVED-X15: ldr x15, [sp
; CHECK-SAVED-X18: ldr x18, [sp

  ret void
}

; Check that caller doesn't shy away from allocating additional CSRs.
define dso_local void @caller() {
; CHECK-LABEL: caller

  %val = load volatile [30 x i64], [30 x i64]* @var
; CHECK-SAVED-X8: adrp x8, var
; CHECK-SAVED-X9: adrp x9, var
; CHECK-SAVED-X10: adrp x10, var
; CHECK-SAVED-X11: adrp x11, var
; CHECK-SAVED-X12: adrp x12, var
; CHECK-SAVED-X13: adrp x13, var
; CHECK-SAVED-X14: adrp x14, var
; CHECK-SAVED-X15: adrp x15, var
; CHECK-SAVED-X18: adrp x18, var

; CHECK-SAVED-ALL: adrp x8, var
; CHECK-SAVED-ALL-DAG: ldr x9
; CHECK-SAVED-ALL-DAG: ldr x10
; CHECK-SAVED-ALL-DAG: ldr x11
; CHECK-SAVED-ALL-DAG: ldr x12
; CHECK-SAVED-ALL-DAG: ldr x13
; CHECK-SAVED-ALL-DAG: ldr x14
; CHECK-SAVED-ALL-DAG: ldr x15
; CHECK-SAVED-ALL-DAG: ldr x18

; CHECK-SAVED-ALL-GISEL: adrp x16, var
; CHECK-SAVED-ALL-GISEL-DAG: ldr x8
; CHECK-SAVED-ALL-GISEL-DAG: ldr x9
; CHECK-SAVED-ALL-GISEL-DAG: ldr x10
; CHECK-SAVED-ALL-GISEL-DAG: ldr x11
; CHECK-SAVED-ALL-GISEL-DAG: ldr x12
; CHECK-SAVED-ALL-GISEL-DAG: ldr x13
; CHECK-SAVED-ALL-GISEL-DAG: ldr x14
; CHECK-SAVED-ALL-GISEL-DAG: ldr x15
; CHECK-SAVED-ALL-GISEL-DAG: ldr x18

  call void @callee()
; CHECK: bl callee

  store volatile [30 x i64] %val, [30 x i64]* @var
; CHECK-SAVED-ALL-DAG: str x9
; CHECK-SAVED-ALL-DAG: str x10
; CHECK-SAVED-ALL-DAG: str x11
; CHECK-SAVED-ALL-DAG: str x12
; CHECK-SAVED-ALL-DAG: str x13
; CHECK-SAVED-ALL-DAG: str x14
; CHECK-SAVED-ALL-DAG: str x15
; CHECK-SAVED-ALL-DAG: str x18

  ret void
}
