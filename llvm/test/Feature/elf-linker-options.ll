; RUN: llc -mtriple x86_64-elf -filetype asm -o - %s | FileCheck %s
; REQUIRES: x86-registered-target

!llvm.linker.options = !{!0, !1}

!0 = !{!"option 0", !"value 0"}
!1 = !{!"option 1", !"value 1"}

; CHECK: .section ".linker-options","e",@llvm_linker_options
; CHECK-NEXT: .ascii  "option 0"
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .ascii  "value 0"
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .ascii  "option 1"
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .ascii  "value 1"
; CHECK-NEXT: .byte   0
