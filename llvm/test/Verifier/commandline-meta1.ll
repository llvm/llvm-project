; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
; Verify that llvm.commandline is properly structured.
; llvm.commandline takes a list of metadata entries.
; Each metadata entry can have only one string.

!llvm.commandline = !{!0}
!0 = !{!"string1", !"string2"}
; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: incorrect number of operands in llvm.commandline metadata
; CHECK-NEXT: !0
