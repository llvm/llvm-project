; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

; Check that we give an error for expressions that are required to be the last
; in the list.

!named = !{!0, !1, !2, !3, !4, !5}
; CHECK: invalid expression
; CHECK-NEXT: !DIExpression({{.+}}, 0, 10, {{.+}})
!0 = !DIExpression(DW_OP_LLVM_fragment, 0, 10, DW_OP_deref)
; CHECK: invalid expression
; CHECK-NEXT: !DIExpression({{.+}}, 0, 20, {{.+}})
!1 = !DIExpression(DW_OP_bit_piece, 0, 20, DW_OP_deref)
; CHECK: invalid expression
; CHECK-NEXT: !DIExpression({{.+}}, 0, 30, {{.+}}, 3, 0)
!2 = !DIExpression(DW_OP_LLVM_fragment, 0, 30, DW_OP_bit_piece, 3, 0)
; CHECK: invalid expression
; CHECK-NEXT: !DIExpression({{.+}}, 40, 0, {{.+}}, 0, 4)
!3 = !DIExpression(DW_OP_bit_piece, 40, 0, DW_OP_LLVM_fragment, 0, 4)

; We shouldn't give an error if other expressions come earlier
; CHECK-NOT: invalid expression
!4 = !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 50)
!5 = !DIExpression(DW_OP_deref, DW_OP_bit_piece, 0, 60)
