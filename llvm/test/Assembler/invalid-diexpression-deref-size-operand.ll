; The size operand of DW_OP_deref_size and DW_OP_xderef_size is encoded as a
; single byte in DWARF, so larger values must be rejected rather than silently
; truncated when emitted.

; RUN: split-file %s %t
; RUN: not llvm-as -disable-output < %t/deref-size.ll 2>&1 | FileCheck %s
; RUN: not llvm-as -disable-output < %t/xderef-size.ll 2>&1 | FileCheck %s

; CHECK: assembly parsed, but does not verify

;--- deref-size.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_deref_size, 256)

;--- xderef-size.ll
!named = !{!0}
!0 = !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef_size, 256)
