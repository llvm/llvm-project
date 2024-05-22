; RUN: llc -mtriple=arm64-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin < %s | FileCheck %s --check-prefix=CHECK-MACHO

; Verify that llvm.commandline metadata is emitted to the corresponding command line section.

; CHECK:              .text
; CHECK:              .section .GCC.command.line,"MS",@progbits,1
; CHECK-NEXT:         .zero 1
; CHECK-NEXT:         .ascii "clang -command1"
; CHECK-NEXT:         .zero 1
; CHECK-NEXT:         .ascii "clang -command2"
; CHECK-NEXT:         .zero 1

; CHECK-MACHO:        .section	__TEXT,__text,regular,pure_instructions
; CHECK-MACHO-NEXT:   .section	__TEXT,__command_line
; CHECK-MACHO-NEXT:   .space	1
; CHECK-MACHO-NEXT:   .ascii	"clang -command1"
; CHECK-MACHO-NEXT:   .space	1
; CHECK-MACHO-NEXT:   .ascii	"clang -command2"
; CHECK-MACHO-NEXT:   .space	1

!llvm.commandline = !{!0, !1}
!0 = !{!"clang -command1"}
!1 = !{!"clang -command2"}
