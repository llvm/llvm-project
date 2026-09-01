# RUN: llvm-mc -triple amdgpu11.00-amd-amdhsa %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple amdgpu11.00-amd-amdhsa %s | \
# RUN:   llvm-dwarfdump -debug-frame - | FileCheck %s --check-prefix=FRAME

# REQUIRES: amdgpu-registered-target

# ASM: .cfi_llvm_def_cfa_constant_address 0, 6
# ASM: .cfi_llvm_def_cfa_register_address_transform 64, 4, 6, 6

.text
.cfi_sections .debug_frame

constant_address:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_constant_address 0, 6
  s_nop 0
  .cfi_endproc

# FRAME: DW_CFA_def_cfa_expression: DW_OP_lit0, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

register_address_transform:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_register_address_transform 64, 4, 6, 6
  s_nop 0
  .cfi_endproc

# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_shl, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address
