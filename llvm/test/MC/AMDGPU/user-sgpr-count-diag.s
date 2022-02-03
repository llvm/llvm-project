// RUN: not llvm-mc --amdhsa-code-object-version=3 -triple amdgcn-amd-amdhsa -mcpu=gfx810 %s 2>&1 >/dev/null | FileCheck -check-prefix=ERR %s

.amdhsa_kernel implied_count_too_low_0
  .amdhsa_user_sgpr_count 0
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
// ERR: [[@LINE+1]]:19: error: amdgpu_user_sgpr_count smaller than than implied by enabled user SGPRs
.end_amdhsa_kernel

.amdhsa_kernel implied_count_too_low_1
  .amdhsa_user_sgpr_count 1
  .amdhsa_user_sgpr_queue_ptr 1
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
// ERR: [[@LINE+1]]:19: error: amdgpu_user_sgpr_count smaller than than implied by enabled user SGPRs
.end_amdhsa_kernel
