; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
; Verify that llvm.commandline is properly structured.
; llvm.commandline takes a list of metadata entries.
; Each metadata entry can contain one string only.

!llvm.commandline = !{!0}
!0 = !{!{!"nested metadata"}}
; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: invalid value for llvm.commandline metadata entry operand(the operand should be a string)
; CHECK-NEXT: !1
