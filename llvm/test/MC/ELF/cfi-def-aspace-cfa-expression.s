# RUN: llvm-mc -triple amdgpu11.00-amd-amdhsa %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple amdgpu11.00-amd-amdhsa %s | \
# RUN:   llvm-dwarfdump -debug-frame - | FileCheck %s --check-prefix=FRAME

# REQUIRES: amdgpu-registered-target

# ASM: .cfi_llvm_def_cfa_address_constant 0, 6
# ASM: .cfi_llvm_def_cfa_address_scaled 64, 4, 64, 6

.text
.cfi_sections .debug_frame

constant_address:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_constant 0, 6
  s_nop 0
  .cfi_endproc

# FRAME: DW_CFA_def_cfa_expression: DW_OP_lit0, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

register_address_transform:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_scaled 64, 4, 64, 6
  s_nop 0
  .cfi_endproc

# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_shl, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

scale_factors:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_scaled 64, 4, 0, 6
# ASM: .cfi_llvm_def_cfa_address_scaled 64, 4, 0, 6
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit0, DW_OP_mul, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_scaled 64, 4, 1, 6
# ASM: .cfi_llvm_def_cfa_address_scaled 64, 4, 1, 6
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_scaled 64, 4, 3, 6
# ASM: .cfi_llvm_def_cfa_address_scaled 64, 4, 3, 6
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit3, DW_OP_mul, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_scaled 64, 4, 32, 6
# ASM: .cfi_llvm_def_cfa_address_scaled 64, 4, 32, 6
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit5, DW_OP_shl, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_scaled 64, 4, 2147483648, 6
# ASM: .cfi_llvm_def_cfa_address_scaled 64, 4, 2147483648, 6
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit31, DW_OP_shl, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_scaled 64, 4, 4294967295, 6
# ASM: .cfi_llvm_def_cfa_address_scaled 64, 4, 4294967295, 6
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_constu 0xffffffff, DW_OP_mul, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_endproc
