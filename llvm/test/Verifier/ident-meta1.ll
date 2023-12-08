; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
; Verify that llvm.ident is properly structured.
; llvm.ident takes a list of metadata entries.
; Each metadata entry can have only one string.

!llvm.ident = !{!0, !1}
!0 = !{!"version string"}
!1 = !{!"string1", !"string2"}
; CHECK: assembly parsed, but does not verify as correct!
; CHECK-NEXT: incorrect number of operands in llvm.ident metadata
; CHECK-NEXT: !1

