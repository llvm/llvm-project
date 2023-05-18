; RUN: llc -mtriple=aarch64-windows -verify-machineinstrs %s -o - \
; RUN:  | FileCheck -check-prefix CHECK-DEFAULT-CODE-MODEL %s
; RUN: llc < %s -mtriple=aarch64-windows -stop-after=prologepilog \
; RUN:  | FileCheck -check-prefix CHECK-REGSTATE %s

; RUN: llc -mtriple=aarch64-windows -verify-machineinstrs -code-model=large %s -o - \
; RUN:  | FileCheck -check-prefix CHECK-LARGE-CODE-MODEL %s
; RUN: llc < %s -mtriple=aarch64-windows -stop-after=prologepilog -code-model=large \
; RUN:  | FileCheck -check-prefix CHECK-REGSTATE-LARGE %s

define void @check_watermark() {
entry:
  %buffer = alloca [4096 x i8], align 1
  ret void
}

; CHECK-DEFAULT-CODE-MODEL: check_watermark:
; CHECK-DEFAULT-CODE-MODEL-DAG: stp x29, x30, [sp
; CHECK-DEFAULT-CODE-MODEL-DAG: mov x15, #256
; CHECK-DEFAULT-CODE-MODEL:     bl __chkstk
; CHECK-DEFAULT-CODE-MODEL:     sub sp, sp, x15, lsl #4

; CHECK-REGSTATE: frame-setup BL &__chkstk, implicit-def $lr, implicit $sp, implicit $x15, implicit-def dead $x16, implicit-def dead $x17, implicit-def dead $nzcv

; CHECK-LARGE-CODE-MODEL: check_watermark:
; CHECK-LARGE-CODE-MODEL-DAG: stp x29, x30, [sp
; CHECK-LARGE-CODE-MODEL-DAG: mov x15, #256
; CHECK-LARGE-CODE-MODEL-DAG: adrp x16, __chkstk
; CHECK-LARGE-CODE-MODEL-DAG: add x16, x16, __chkstk
; CHECK-LARGE-CODE-MODEL:     blr x16
; CHECK-LARGE-CODE-MODEL:     sub sp, sp, x15, lsl #4

; CHECK-REGSTATE-LARGE: frame-setup BLR killed $x16, implicit-def $lr, implicit $sp, implicit-def $x15, implicit-def dead $x16, implicit-def dead $x17, implicit-def dead $nzcv
