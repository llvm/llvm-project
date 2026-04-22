// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm %s 2>&1 | FileCheck %s

.amdhsa_code_object_version 6

 // CHECK:.amdhsa_user_sgpr_count 33
.amdhsa_kernel many_inreg_i32
        .amdhsa_next_free_vgpr 0
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 33

.end_amdhsa_kernel
