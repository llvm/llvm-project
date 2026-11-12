; RUN: not opt -S < %s 2>&1 | FileCheck %s

; We can only use the internal variant of the entry value operation,
; DW_OP_LLVM_entry_value, in DIExpressions.

!named = !{!0}
; CHECK: invalid expression
!0 = !DIExpression(DW_OP_entry_value, 1)
