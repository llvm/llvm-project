// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR %s

.amdhsa_code_object_version 6

.amdhsa_kernel user_sgpr_number_out_of_bits
        .amdhsa_next_free_vgpr 0
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 64
        // GFX1250-ERR: too many user SGPRs enabled
.end_amdhsa_kernel

.amdhsa_kernel user_sgpr_number_out_of_bits
        .amdhsa_next_free_vgpr 0
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 33
        // GFX1250-ERR: number of user sgpr count is out of bound
.end_amdhsa_kernel
