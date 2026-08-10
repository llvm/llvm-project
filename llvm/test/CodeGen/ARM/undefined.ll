; RUN: llc -mtriple armv7-eabi -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv6m-eabi -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-eabi -o - %s | FileCheck %s

declare void @llvm.arm.undefined(i32) nounwind

define void @undefined_trap() {
entry:
  tail call void @llvm.arm.undefined(i32 254)
  ret void
}

; CHECK-LABEL: undefined_trap
; CHECK: udf #254
