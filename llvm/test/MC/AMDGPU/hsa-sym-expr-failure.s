// RUN: not llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefix=ASM %s

// Some expression currently require (immediately) solvable expressions, i.e.,
// they don't depend on yet-unknown symbolic values.

.text

.amdhsa_code_object_version 4

.p2align 8
.type user_sgpr_count,@function
user_sgpr_count:
  s_endpgm

.p2align 6
.amdhsa_kernel user_sgpr_count
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_count defined_boolean
.end_amdhsa_kernel


.p2align 8
.type user_sgpr_private_segment_buffer,@function
user_sgpr_private_segment_buffer:
  s_endpgm

.amdhsa_kernel user_sgpr_private_segment_buffer
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_private_segment_buffer defined_boolean
.end_amdhsa_kernel

.p2align 8
.type user_sgpr_kernarg_preload_length,@function
user_sgpr_kernarg_preload_length:
  s_endpgm

.amdhsa_kernel user_sgpr_kernarg_preload_length
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_kernarg_preload_length defined_boolean
.end_amdhsa_kernel

.p2align 8
.type user_sgpr_kernarg_preload_offset,@function
user_sgpr_kernarg_preload_offset:
  s_endpgm

.amdhsa_kernel user_sgpr_kernarg_preload_offset
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_kernarg_preload_offset defined_boolean
.end_amdhsa_kernel

.p2align 8
.type user_sgpr_dispatch_ptr,@function
user_sgpr_dispatch_ptr:
  s_endpgm

.p2align 6
.amdhsa_kernel user_sgpr_dispatch_ptr
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_dispatch_ptr defined_boolean
.end_amdhsa_kernel

.p2align 8
.type user_sgpr_queue_ptr,@function
user_sgpr_queue_ptr:
  s_endpgm

.p2align 6
.amdhsa_kernel user_sgpr_queue_ptr
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_queue_ptr defined_boolean
.end_amdhsa_kernel

.p2align 8
.type user_sgpr_kernarg_segment_ptr,@function
user_sgpr_kernarg_segment_ptr:
  s_endpgm

.p2align 6
.amdhsa_kernel user_sgpr_kernarg_segment_ptr
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_kernarg_segment_ptr defined_boolean
.end_amdhsa_kernel

.p2align 8
.type user_sgpr_dispatch_id,@function
user_sgpr_dispatch_id:
  s_endpgm

.p2align 6
.amdhsa_kernel user_sgpr_dispatch_id
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_dispatch_id defined_boolean
.end_amdhsa_kernel

.p2align 8
.type user_sgpr_flat_scratch_init,@function
user_sgpr_flat_scratch_init:
  s_endpgm

.p2align 6
.amdhsa_kernel user_sgpr_flat_scratch_init
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_flat_scratch_init defined_boolean
.end_amdhsa_kernel

.p2align 8
.type user_sgpr_private_segment_size,@function
user_sgpr_private_segment_size:
  s_endpgm

.p2align 6
.amdhsa_kernel user_sgpr_private_segment_size
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_user_sgpr_private_segment_size defined_boolean
.end_amdhsa_kernel

.p2align 8
.type wavefront_size32,@function
wavefront_size32:
  s_endpgm

.p2align 6
.amdhsa_kernel wavefront_size32
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_wavefront_size32 defined_boolean
.end_amdhsa_kernel

.p2align 8
.type next_free_vgpr,@function
next_free_vgpr:
  s_endpgm

.p2align 6
.amdhsa_kernel next_free_vgpr
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_next_free_vgpr defined_boolean
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

.p2align 8
.type next_free_sgpr,@function
next_free_sgpr:
  s_endpgm

.p2align 6
.amdhsa_kernel next_free_sgpr
  .amdhsa_next_free_vgpr 0
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_next_free_sgpr defined_boolean
  .amdhsa_accum_offset 4
.end_amdhsa_kernel

.p2align 8
.type accum_offset,@function
accum_offset:
  s_endpgm

.p2align 6
.amdhsa_kernel accum_offset
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_accum_offset defined_boolean
.end_amdhsa_kernel

.p2align 8
.type reserve_vcc,@function
reserve_vcc:
  s_endpgm

.p2align 6
.amdhsa_kernel reserve_vcc
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_reserve_vcc defined_boolean
.end_amdhsa_kernel

.p2align 8
.type reserve_flat_scratch,@function
reserve_flat_scratch:
  s_endpgm

.p2align 6
.amdhsa_kernel reserve_flat_scratch
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_reserve_flat_scratch defined_boolean
.end_amdhsa_kernel

.p2align 8
.type shared_vgpr_count,@function
shared_vgpr_count:
  s_endpgm

.p2align 6
.amdhsa_kernel shared_vgpr_count
  .amdhsa_next_free_vgpr 0
  .amdhsa_next_free_sgpr 0
  .amdhsa_accum_offset 4
// ASM: :[[@LINE+1]]:{{[0-9]+}}: error: directive should have resolvable expression
  .amdhsa_shared_vgpr_count defined_boolean
.end_amdhsa_kernel

.set defined_boolean, 1
