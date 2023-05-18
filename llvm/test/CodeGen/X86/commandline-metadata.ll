; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s
; Verify that llvm.commandline metadata is emitted to a section named
; .GCC.command.line with each line separated with null bytes.

; CHECK: .section .GCC.command.line,"MS",@progbits,1
; CHECK-NEXT: .zero 1
; CHECK-NEXT: .ascii "clang -command -line"
; CHECK-NEXT: .zero 1
; CHECK-NEXT: .ascii "something else"
; CHECK-NEXT: .zero 1
!llvm.commandline = !{!0, !1}
!0 = !{!"clang -command -line"}
!1 = !{!"something else"}
