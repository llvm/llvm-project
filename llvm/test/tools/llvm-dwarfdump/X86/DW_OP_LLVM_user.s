# RUN: llvm-mc %s -filetype=obj -triple=i686-pc-linux -o %t
# RUN: llvm-dwarfdump -v %t | FileCheck %s

# FIXME: Is there a better approach than using `DW_CFA_expression EAX <op>`?

# CHECK:      .eh_frame contents:
# CHECK:      FDE
# CHECK-NEXT: Format: DWARF32

foo:
 .cfi_startproc
 # CHECK-NEXT: DW_CFA_expression: EAX <decoding error> e9 00
 .cfi_escape 0x10, 0x00, 0x02, 0xe9, 0x00
 # CHECK-NEXT: DW_CFA_expression: EAX DW_OP_LLVM_user DW_OP_LLVM_nop
 .cfi_escape 0x10, 0x00, 0x02, 0xe9, 0x01
 .cfi_endproc
