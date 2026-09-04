; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; Label IDs are uint64_t values, so use the largest one while checking that the
; opcodes and IDs survive two bitcode round-trips.

!named = !{!0}

; CHECK: !DIExpression(DW_OP_LLVM_label, 18446744073709551615, DW_OP_LLVM_bra, 7, DW_OP_LLVM_skip, 18446744073709551615, DW_OP_LLVM_label, 7)
!0 = !DIExpression(DW_OP_LLVM_label, 18446744073709551615,
                   DW_OP_LLVM_bra, 7,
                   DW_OP_LLVM_skip, 18446744073709551615,
                   DW_OP_LLVM_label, 7)
