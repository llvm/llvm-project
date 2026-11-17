; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
;
; Symbolic and numeric sub-opcodes round-trip to the same named operation.
; CHECK: !named = !{!DIExpression(DW_OP_constu, 6, DW_OP_LLVM_user, DW_OP_LLVM_form_aspace_address), !DIExpression(DW_OP_constu, 6, DW_OP_LLVM_user, DW_OP_LLVM_form_aspace_address), !DIExpression(DW_OP_constu, 6, DW_OP_LLVM_user, DW_OP_LLVM_form_aspace_address), !DIExpression(DW_OP_constu, 233, DW_OP_stack_value)}

!named = !{!0, !1, !2, !3}
!0 = !DIExpression(DW_OP_constu, 6, DW_OP_LLVM_user, DW_OP_LLVM_form_aspace_address)
!1 = !DIExpression(DW_OP_constu, 6, DW_OP_LLVM_user, 2)
!2 = !DIExpression(16, 6, 233, 2)
; The value 233 is an operand here, so the next token is a regular opcode.
!3 = !DIExpression(DW_OP_constu, 233, DW_OP_stack_value)
