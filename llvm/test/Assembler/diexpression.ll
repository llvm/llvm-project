; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6}
!named = !{!0, !1, !2, !3, !4, !5, !6}

; CHECK:      !0 = !DIExpression()
; CHECK-NEXT: !1 = !DIExpression(DW_OP_deref)
; CHECK-NEXT: !2 = !DIExpression(DW_OP_constu, 3, DW_OP_plus)
; CHECK-NEXT: !3 = !DIExpression(DW_OP_LLVM_fragment, 3, 7)
; CHECK-NEXT: !4 = !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 3, DW_OP_LLVM_fragment, 3, 7)
; CHECK-NEXT: !5 = !DIExpression(DW_OP_constu, 2, DW_OP_swap, DW_OP_xderef)
; CHECK-NEXT: !6 = !DIExpression(DW_OP_plus_uconst, 3)
!0 = !DIExpression()
!1 = !DIExpression(DW_OP_deref)
!2 = !DIExpression(DW_OP_constu, 3, DW_OP_plus)
!3 = !DIExpression(DW_OP_LLVM_fragment, 3, 7)
!4 = !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 3, DW_OP_LLVM_fragment, 3, 7)
!5 = !DIExpression(DW_OP_constu, 2, DW_OP_swap, DW_OP_xderef)
!6 = !DIExpression(DW_OP_plus_uconst, 3)
