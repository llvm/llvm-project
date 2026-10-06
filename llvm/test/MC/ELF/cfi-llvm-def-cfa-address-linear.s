# RUN: llvm-mc -triple amdgpu11.00-amd-amdhsa %s | \
# RUN:   llvm-mc -triple amdgpu11.00-amd-amdhsa | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple amdgpu11.00-amd-amdhsa %s | \
# RUN:   llvm-dwarfdump -debug-frame - | FileCheck %s --check-prefix=FRAME
# RUN: llvm-mc -filetype=obj -triple amdgpu11.00-amd-amdhsa %s | \
# RUN:   llvm-readobj --hex-dump=.debug_frame - | FileCheck %s --check-prefix=BYTES

# REQUIRES: amdgpu-registered-target

# Preserve the constant-zero, wave64, and wave32 CFA expression bytes.
# BYTES:      0x{{[0-9a-f]+}} {{.*}} 410f0430 36e90200
# BYTES:      0x{{[0-9a-f]+}} {{.*}} 410f0990 40940436
# BYTES-NEXT: 0x{{[0-9a-f]+}} 2436e902
# BYTES:      0x{{[0-9a-f]+}} 410f0990 40940435 2436e902

.text
.cfi_sections .debug_frame

constant_address:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0{{$}}
# FRAME: DW_CFA_def_cfa_expression: DW_OP_lit0, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_endproc

wave64_address:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 64
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 64
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_shl, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_endproc

wave32_address:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 32
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 32
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit5, DW_OP_shl, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_endproc

scale_factors:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 0
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 0
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit0, DW_OP_mul, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 1
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 1
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 3
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 3
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit3, DW_OP_mul, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 32
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 32
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit5, DW_OP_shl, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 2147483648
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 2147483648
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit31, DW_OP_shl, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 4294967295
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 4294967295
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_constu 0xffffffff, DW_OP_mul, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_endproc

signed_offsets:
  .cfi_startproc
  .cfi_llvm_def_cfa_address_linear 6, 17
# ASM: .cfi_llvm_def_cfa_address_linear 6, 17{{$}}
# FRAME: DW_CFA_def_cfa_expression: DW_OP_lit17, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, -17
# ASM: .cfi_llvm_def_cfa_address_linear 6, -17{{$}}
# FRAME: DW_CFA_def_cfa_expression: DW_OP_consts -17, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 17, 64, 4, 3
# ASM: .cfi_llvm_def_cfa_address_linear 6, 17, 64, 4, 3
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit3, DW_OP_mul, DW_OP_plus_uconst 0x11, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, -17, 64, 4, 32
# ASM: .cfi_llvm_def_cfa_address_linear 6, -17, 64, 4, 32
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit5, DW_OP_shl, DW_OP_consts -17, DW_OP_plus, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 17, 64, 4, 0
# ASM: .cfi_llvm_def_cfa_address_linear 6, 17, 64, 4, 0
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit0, DW_OP_mul, DW_OP_plus_uconst 0x11, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 9223372036854775807
# ASM: .cfi_llvm_def_cfa_address_linear 6, 9223372036854775807{{$}}
# FRAME: DW_CFA_def_cfa_expression: DW_OP_constu 0x7fffffffffffffff, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, -9223372036854775808
# ASM: .cfi_llvm_def_cfa_address_linear 6, -9223372036854775808{{$}}
# FRAME: DW_CFA_def_cfa_expression: DW_OP_consts -9223372036854775808, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 9223372036854775807, 64, 4, 1
# ASM: .cfi_llvm_def_cfa_address_linear 6, 9223372036854775807, 64, 4, 1
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_plus_uconst 0x7fffffffffffffff, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, -9223372036854775808, 64, 4, 1
# ASM: .cfi_llvm_def_cfa_address_linear 6, -9223372036854775808, 64, 4, 1
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_consts -9223372036854775808, DW_OP_plus, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_endproc

high_bit_offset_aliases:
  .cfi_startproc
  .cfi_llvm_def_cfa_address_linear 6, 0xffffffffffffffff
# ASM: .cfi_llvm_def_cfa_address_linear 6, -1{{$}}
# FRAME: DW_CFA_def_cfa_expression: DW_OP_consts -1, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0x8000000000000000, 64, 4, 1
# ASM: .cfi_llvm_def_cfa_address_linear 6, -9223372036854775808, 64, 4, 1
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_consts -9223372036854775808, DW_OP_plus, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_endproc

unsigned_boundaries:
  .cfi_startproc
  .cfi_llvm_def_cfa_address_linear 6, 0, s32, 4, 1
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 64, 4, 1
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 0, 0, 0, 1, 1
# ASM: .cfi_llvm_def_cfa_address_linear 0, 0, 0, 1, 1
# FRAME: DW_CFA_def_cfa_expression: DW_OP_reg0, DW_OP_deref_size 0x1, DW_OP_lit0, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 4294967295, 0, 2147483648, 255, 1
# ASM: .cfi_llvm_def_cfa_address_linear 4294967295, 0, 2147483648, 255, 1
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx 0x80000000, DW_OP_deref_size 0xff, DW_OP_constu 0xffffffff, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 0, 4294967295, 4, 1
# ASM: .cfi_llvm_def_cfa_address_linear 6, 0, 4294967295, 4, 1
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx 0xffffffff, DW_OP_deref_size 0x4, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address

  s_nop 0
  .cfi_endproc

constant_128:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 128
# ASM: .cfi_llvm_def_cfa_address_linear 6, 128{{$}}
# FRAME: DW_CFA_def_cfa_expression: DW_OP_constu 0x80, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address
  .cfi_endproc

scaled_offset_128:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_def_cfa_address_linear 6, 128, 64, 4, 32
# ASM: .cfi_llvm_def_cfa_address_linear 6, 128, 64, 4, 32{{$}}
# FRAME: DW_CFA_def_cfa_expression: DW_OP_regx SGPR32, DW_OP_deref_size 0x4, DW_OP_lit5, DW_OP_shl, DW_OP_plus_uconst 0x80, DW_OP_lit6, DW_OP_LLVM_user DW_OP_LLVM_form_aspace_address
  .cfi_endproc
