// RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx904 --amdhsa-code-object-version=3 -mattr=+xnack < %s | FileCheck --check-prefix=ASM %s

.text
// ASM: .text

.amdgcn_target "amdgcn-amd-amdhsa--gfx904+xnack"
// ASM: .amdgcn_target "amdgcn-amd-amdhsa--gfx904+xnack"


// ASM-LABEL: .amdhsa_kernel user_sgprs_implied_count
// ASM: .amdhsa_user_sgpr_count 15
.amdhsa_kernel user_sgprs_implied_count_all
  .amdhsa_user_sgpr_private_segment_buffer 1
  .amdhsa_user_sgpr_dispatch_ptr 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_id 1
  .amdhsa_user_sgpr_flat_scratch_init 1
  .amdhsa_user_sgpr_private_segment_size 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

// ASM-LABEL: .amdhsa_kernel user_sgprs_implied_count_0
// ASM: .amdhsa_user_sgpr_count 7
.amdhsa_kernel user_sgprs_implied_count_0
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_flat_scratch_init 1
  .amdhsa_user_sgpr_private_segment_size 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

// ASM-LABEL: .amdhsa_kernel user_sgprs_implied_count_1
// ASM: .amdhsa_user_sgpr_count 9
.amdhsa_kernel user_sgprs_implied_count_1
  .amdhsa_user_sgpr_private_segment_buffer 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_private_segment_size 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel


// ASM-LABEL: .amdhsa_kernel user_sgprs_implied_count_private_segment_buffer
// ASM: .amdhsa_user_sgpr_count 4
  .amdhsa_kernel user_sgprs_implied_count_private_segment_buffer
  .amdhsa_user_sgpr_private_segment_buffer 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel


// ASM-LABEL: .amdhsa_kernel explicit_user_sgpr_count_16
.amdhsa_kernel explicit_user_sgpr_count_16
  .amdhsa_user_sgpr_count 16
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel


// ASM-LABEL: .amdhsa_kernel explicit_user_sgpr_count_0
// ASM: .amdhsa_user_sgpr_count 0
  .amdhsa_kernel explicit_user_sgpr_count_0
  .amdhsa_user_sgpr_count 0
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

// ASM-LABEL: .amdhsa_kernel explicit_user_sgpr_count_1
// ASM: .amdhsa_user_sgpr_count 1
.amdhsa_kernel explicit_user_sgpr_count_1
  .amdhsa_user_sgpr_count 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel

.amdhsa_kernel explicit_user_sgpr_count_larger_than_implied
  .amdhsa_user_sgpr_count 12
  .amdhsa_user_sgpr_private_segment_buffer 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel
