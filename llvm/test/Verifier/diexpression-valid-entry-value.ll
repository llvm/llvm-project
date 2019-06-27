; RUN: opt -S < %s 2>&1 | FileCheck %s

!named = !{!0}
; CHECK-NOT: invalid expression
!0 = !DIExpression(DW_OP_entry_value, 1)
