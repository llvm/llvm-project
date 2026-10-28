; RUN: not opt -S < %s 2>&1 | FileCheck %s

; Only DW_OP_LLVM_offset is supported as a DW_OP_LLVM_user operation.

!named = !{!0, !1}
; CHECK: invalid expression
; CHECK-NEXT: !DIExpression(233, 2)
!0 = !DIExpression(DW_OP_LLVM_user, 2)
; CHECK: invalid expression
; CHECK-NEXT: !DIExpression(233)
!1 = !DIExpression(DW_OP_LLVM_user)
