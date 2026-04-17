// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR %s

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdhsa_code_object_version 6

.amdhsa_kernel many_inreg_i32
        .amdhsa_next_free_vgpr 0
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 65
// GFX1250-ERR: too many user SGPRs enabled
.end_amdhsa_kernel
