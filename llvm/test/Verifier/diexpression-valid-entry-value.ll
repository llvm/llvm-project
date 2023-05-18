; RUN: opt -S < %s 2>&1 | FileCheck %s

!named = !{!0, !1}
; CHECK-NOT: invalid expression
!0 = !DIExpression(DW_OP_LLVM_entry_value, 1)
!1 = !DIExpression(DW_OP_LLVM_entry_value, 1, DW_OP_lit0, DW_OP_plus)
