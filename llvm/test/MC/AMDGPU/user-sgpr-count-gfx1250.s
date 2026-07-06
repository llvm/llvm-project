// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm %s 2>&1 | FileCheck %s

.amdhsa_code_object_version 6

// CHECK:.amdhsa_user_sgpr_count 8
        .amdhsa_kernel user_sgpr_0
        .amdhsa_next_free_vgpr 0
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 8
.end_amdhsa_kernel

// CHECK:.amdhsa_user_sgpr_count 31
.amdhsa_kernel user_sgpr_1
        .amdhsa_next_free_vgpr 1
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 31
.end_amdhsa_kernel

// CHECK:.amdhsa_user_sgpr_count 32
.amdhsa_kernel user_sgpr_2
        .amdhsa_next_free_vgpr 1
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 32
        .end_amdhsa_kernel
