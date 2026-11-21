; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: error: invalid DW_OP_LLVM_user sub-op 'DW_OP_deref'

!named = !{!0}
!0 = !DIExpression(DW_OP_LLVM_user, DW_OP_deref)
