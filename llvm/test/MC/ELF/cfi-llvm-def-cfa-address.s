# RUN: llvm-mc -triple amdgpu11.00-amd-amdhsa %s | \
# RUN:   llvm-mc -triple amdgpu11.00-amd-amdhsa | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple amdgpu11.00-amd-amdhsa %s -o %t
# RUN: llvm-dwarfdump -debug-frame %t | FileCheck %s --check-prefix=FRAME

# REQUIRES: amdgpu-registered-target

.text
.cfi_sections .debug_frame
f:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_constant 6, 0
# ASM: .cfi_llvm_def_cfa_address_constant 6, 0
# FRAME: DW_CFA_def_cfa_expression: DW_OP_lit0, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address{{$}}

  s_nop 0
  .cfi_llvm_def_cfa_address_scaled 6, 0, 64, 4, 32
# ASM: .cfi_llvm_def_cfa_address_scaled 6, 0, 64, 4, 32
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit5, DW_OP_shl, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address{{$}}

  s_nop 0
  .cfi_llvm_def_cfa_address_constant 0, 0
# ASM: .cfi_llvm_def_cfa_address_constant 0, 0
# FRAME: DW_CFA_def_cfa_expression: DW_OP_lit0{{$}}

  s_nop 0
  .cfi_llvm_def_cfa_address_scaled 0, 0, 16, 8, 1
# ASM: .cfi_llvm_def_cfa_address_scaled 0, 0, 16, 8, 1
# FRAME: DW_CFA_def_cfa_expression: DW_OP_reg16 PC_REG, DW_OP_deref{{$}}

  s_nop 0
  .cfi_endproc
