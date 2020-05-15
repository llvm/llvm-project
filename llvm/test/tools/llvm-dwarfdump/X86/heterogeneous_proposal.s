# XFAIL: *
# RUN: llvm-mc %s -filetype=obj -triple=i686-pc-linux -o %t
# RUN: llvm-dwarfdump -v %t | FileCheck %s

# Check that we can decode new ops described at
# llvm/docs/AMDGPUUsage.rst#expression-operation-encodings

# FIXME: Is there a better approach than using `DW_CFA_expression reg0 <op>`?

# CHECK:      .eh_frame contents:
# CHECK:      FDE

foo:
 .cfi_startproc
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_form_aspace_address
 .cfi_escape 0x10, 0x00, 0x01, 0xe1
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_push_lane
 .cfi_escape 0x10, 0x00, 0x01, 0xe2
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_offset
 .cfi_escape 0x10, 0x00, 0x01, 0xe3
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_offset_uconst 0x0
 .cfi_escape 0x10, 0x00, 0x02, 0xe4, 0x00
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_bit_offset
 .cfi_escape 0x10, 0x00, 0x01, 0xe5
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_call_frame_entry_reg EAX
 .cfi_escape 0x10, 0x00, 0x02, 0xe6, 0x00
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_undefined
 .cfi_escape 0x10, 0x00, 0x01, 0xe7
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_aspace_bregx EAX+2
 .cfi_escape 0x10, 0x00, 0x03, 0xe8, 0x0, 0x2
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_aspace_implicit_pointer 0x1 +2
 .cfi_escape 0x10, 0x00, 0x06, 0xe9, 0x1, 0x0, 0x0, 0x0, 0x2
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_piece_end
 .cfi_escape 0x10, 0x00, 0x01, 0xea
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_extend 0x0 0x0
 .cfi_escape 0x10, 0x00, 0x03, 0xeb, 0x0, 0x0
 # CHECK-NEXT: DW_CFA_expression: reg0 DW_OP_LLVM_select_bit_piece 0x0 0x0
 .cfi_escape 0x10, 0x00, 0x03, 0xec, 0x0, 0x0
 .cfi_endproc

# CHECK-NEXT:     DW_CFA_nop:
