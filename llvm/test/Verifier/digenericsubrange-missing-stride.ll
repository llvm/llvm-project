; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

!named = !{!0}
; CHECK: GenericSubrange must contain stride
!0 = !DIGenericSubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 80, DW_OP_plus, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 120, DW_OP_plus, DW_OP_deref))
