// RUN: not llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx1310 -filetype=null %s 2>&1 | FileCheck --check-prefix=ERR %s

.text

.amdhsa_kernel complete
// ERR: error: too many user SGPRs enabled, found 33, but only 32 are supported.
  .amdhsa_user_sgpr_count 33

  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 27
.end_amdhsa_kernel

