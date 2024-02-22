; RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s | llvm-dwarfdump -debug-frame - | FileCheck %s

.text
.cfi_sections .debug_frame

; CHECK-NOT: DW_CFA_expression

register_pair:
  .cfi_startproc
  s_nop 2
  ; CHECK: DW_CFA_expression: PC_REG DW_OP_regx SGPR30_LO16, DW_OP_piece 0x4, DW_OP_regx SGPR31_LO16, DW_OP_piece 0x4
  .cfi_llvm_register_pair 16, 62, 32, 63, 32
  s_nop 2
  .cfi_endproc

; CHECK-NOT: DW_CFA_expression

vector_registers:
  .cfi_startproc
  s_nop 2
  ; CHECK: DW_CFA_expression: PC_REG DW_OP_regx 0x67f, DW_OP_bit_piece 0x20 0x0, DW_OP_regx 0x67f, DW_OP_bit_piece 0x20 0x20
  .cfi_llvm_vector_registers 16, 1663, 0, 32, 1663, 1, 32
  s_nop 2
  .cfi_endproc

; CHECK-NOT: DW_CFA_expression

vector_registers_single:
  .cfi_startproc
  s_nop 2
  ;; Note that 0x2c below is the offset in the VGPR, so 4 (bytes, vgpr lane size) * 11 (the lane).
  ; CHECK: DW_CFA_expression: SGPR45_LO16 DW_OP_regx VGPR41_LO16, DW_OP_LLVM_user DW_OP_LLVM_offset_uconst 0x2c
  .cfi_llvm_vector_registers 77, 2601, 11, 32
  s_nop 2
  .cfi_endproc

; CHECK-NOT: DW_CFA_expression

vector_offsets:
  .cfi_startproc
  s_nop 2
  ; CHECK: DW_CFA_expression: VGPR40_LO16 DW_OP_regx VGPR40_LO16, DW_OP_swap, DW_OP_LLVM_user DW_OP_LLVM_offset_uconst 0x100, DW_OP_LLVM_user DW_OP_LLVM_call_frame_entry_reg EXEC, DW_OP_deref_size 0x8, DW_OP_LLVM_user DW_OP_LLVM_select_bit_piece 0x20 0x40
  .cfi_llvm_vector_offset 2600, 32, 17, 64, 256
  s_nop 2
  .cfi_endproc

; CHECK-NOT: DW_CFA_expression
