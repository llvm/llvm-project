; RUN: llc -mtriple armv7-unknown -frame-pointer=all -filetype=asm -o - %s | FileCheck %s --check-prefix=CHECK-NO-CFI
; RUN: llc -mtriple armv7-unknown -frame-pointer=all -filetype=asm -force-dwarf-frame-section -o - %s | FileCheck %s --check-prefix=CHECK-ALWAYS-CFI

declare void @dummy_use(i32*, i32)

define void @test_basic() #0 {
        %mem = alloca i32, i32 10
        call void @dummy_use (i32* %mem, i32 10)
  ret void
}

; CHECK-NO-CFI-LABEL: test_basic:
; CHECK-NO-CFI:   .fnstart
; CHECK-NO-CFI-NOT:   .cfi_sections .debug_frame
; CHECK-NO-CFI-NOT:   .cfi_startproc
; CHECK-NO-CFI:       @ %bb.0:
; CHECK-NO-CFI:       push {r11, lr}
; CHECK-NO-CFI-NOT:   .cfi_def_cfa_offset 8
; CHECK-NO-CFI-NOT:   .cfi_offset lr, -4
; CHECK-NO-CFI-NOT:   .cfi_offset r11, -8
; CHECK-NO-CFI:       mov r11, sp
; CHECK-NO-CFI-NOT:   .cfi_def_cfa_register r11
; CHECK-NO-CFI-NOT:   .cfi_endproc
; CHECK-NO-CFI:       .fnend

; CHECK-ALWAYS-CFI-LABEL: test_basic:
; CHECK-ALWAYS-CFI:   .fnstart
; CHECK-ALWAYS-CFI:   .cfi_sections .debug_frame
; CHECK-ALWAYS-CFI:   .cfi_startproc
; CHECK-ALWAYS-CFI:   @ %bb.0:
; CHECK-ALWAYS-CFI:   push {r11, lr}
; CHECK-ALWAYS-CFI:   .cfi_def_cfa_offset 8
; CHECK-ALWAYS-CFI:   .cfi_offset lr, -4
; CHECK-ALWAYS-CFI:   .cfi_offset r11, -8
; CHECK-ALWAYS-CFI:   mov r11, sp
; CHECK-ALWAYS-CFI:   .cfi_def_cfa_register r11
; CHECK-ALWAYS-CFI:   .cfi_endproc
; CHECK-ALWAYS-CFI:   .fnend
