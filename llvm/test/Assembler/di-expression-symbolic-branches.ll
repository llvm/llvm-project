; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Labels use IDs in a DIExpression, so make sure the assembler keeps both
; forward and backward references.

!named = !{!0}

; CHECK: !DIExpression(DW_OP_LLVM_label, 0, DW_OP_LLVM_bra, 42, DW_OP_LLVM_skip, 0, DW_OP_LLVM_label, 42)
!0 = !DIExpression(DW_OP_LLVM_label, 0,
                   DW_OP_LLVM_bra, 42,
                   DW_OP_LLVM_skip, 0,
                   DW_OP_LLVM_label, 42)
