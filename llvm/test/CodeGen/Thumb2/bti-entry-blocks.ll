; RUN: llc < %s -mtriple=thumbv7m-arm-none-eabi | FileCheck %s

define hidden i32 @linkage_external() local_unnamed_addr "branch-target-enforcement" {
; CHECK-LABEL: linkage_external:
; CHECK: bti
; CHECK-NEXT: movs r0, #1
; CHECK-NEXT: bx lr
entry:
  ret i32 1
}

define internal i32 @linkage_internal() unnamed_addr "branch-target-enforcement" {
; CHECK-LABEL: linkage_internal:
; CHECK: bti
; CHECK: movs r0, #2
; CHECK-NEXT: bx lr
entry:
  ret i32 2
}
