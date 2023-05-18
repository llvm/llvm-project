; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!named = !{!0}
; CHECK: GenericSubrange must contain count or upperBound
!0 = !DIGenericSubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 80, DW_OP_plus, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 112, DW_OP_plus, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
