; DW_OP_xderef_size takes a one-byte size operand. A trailing occurrence with
; no operand must be rejected, which only happens if ExprOperand::getSize()
; accounts for the operand.

; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; CHECK: assembly parsed, but does not verify
!named = !{!0}
!0 = !DIExpression(DW_OP_xderef_size)
