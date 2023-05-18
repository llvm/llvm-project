; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
; Verify that llvm.ident is properly structured.
; llvm.ident takes a list of metadata entries.
; Each metadata entry can contain one string only.

!llvm.ident = !{!0}
!0 = !{!{!"nested metadata"}}
; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: invalid value for llvm.ident metadata entry operand(the operand should be a string)
; CHECK-NEXT: !1
