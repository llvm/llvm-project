; RUN: llc -mtriple x86_64-elf -filetype asm -o - %s | FileCheck %s
; REQUIRES: x86-registered-target

!llvm.dependent-libraries = !{!0, !1, !0}

!0 = !{!"foo"}
!1 = !{!"b a r"}

; CHECK: .section .deplibs,"MS",@llvm_dependent_libraries,1
; CHECK-NEXT: .ascii  "foo"
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .ascii  "b a r"
; CHECK-NEXT: .byte   0
; CHECK-NEXT: .ascii  "foo"
; CHECK-NEXT: .byte   0
