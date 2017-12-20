; RUN: llc -mtriple=aarch64-windows -verify-machineinstrs %s -o - \
; RUN:  | FileCheck -check-prefix CHECK-DEFAULT-CODE-MODEL %s

; RUN: llc -mtriple=aarch64-windows -verify-machineinstrs -code-model=large %s -o - \
; RUN:  | FileCheck -check-prefix CHECK-LARGE-CODE-MODEL %s

define void @check_watermark() {
entry:
  %buffer = alloca [4096 x i8], align 1
  ret void
}

; CHECK-DEFAULT-CODE-MODEL: check_watermark:
; CHECK-DEFAULT-CODE-MODEL-DAG: stp x29, x30, [sp
; CHECK-DEFAULT-CODE-MODEL-DAG: orr x15, xzr, #0x100
; CHECK-DEFAULT-CODE-MODEL:     bl __chkstk
; CHECK-DEFAULT-CODE-MODEL:     sub sp, sp, x15, lsl #4

; CHECK-LARGE-CODE-MODEL: check_watermark:
; CHECK-LARGE-CODE-MODEL-DAG: stp x29, x30, [sp
; CHECK-LARGE-CODE-MODEL-DAG: orr x15, xzr, #0x100
; CHECK-LARGE-CODE-MODEL-DAG: adrp x16, __chkstk
; CHECK-LARGE-CODE-MODEL-DAG: add x16, x16, __chkstk
; CHECK-LARGE-CODE-MODEL:     blr x16
; CHECK-LARGE-CODE-MODEL:     sub sp, sp, x15, lsl #4
