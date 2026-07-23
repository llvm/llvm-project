// RUN: split-file %s %t
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa-llvm -mcpu=gfx802 %t/amdhsa-llvm.s | FileCheck --check-prefix=AMDHSA-LLVM %s
// RUN: llvm-mc -triple=amdgcn-amd-amdpal-llvm -mcpu=gfx802 %t/amdpal-llvm.s | FileCheck --check-prefix=AMDPAL-LLVM %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx802 %t/amdhsa-llvm.s -filetype=null 2>&1 | FileCheck --check-prefix=AMDHSA-ERR %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdpal -mcpu=gfx802 %t/amdpal-llvm.s -filetype=null 2>&1 | FileCheck --check-prefix=AMDPAL-ERR %s

// Test that the environment component of the triple is preserved and validated

//--- amdhsa-llvm.s
// AMDHSA-LLVM: .amdgcn_target "amdgcn-amd-amdhsa-llvm-gfx802"
// AMDHSA-ERR: error: .amdgcn_target amdgcn-amd-amdhsa-llvm-gfx802 is incompatible with amdgcn-amd-amdhsa-unknown-gfx802
.amdgcn_target "amdgcn-amd-amdhsa-llvm-gfx802"

//--- amdpal-llvm.s
// AMDPAL-LLVM: .amd_amdgpu_isa "amdgcn-amd-amdpal-llvm-gfx802"
// AMDPAL-ERR: error: .amd_amdgpu_isa amdgcn-amd-amdpal-llvm-gfx802 is incompatible with amdgcn-amd-amdpal-unknown-gfx802
.amd_amdgpu_isa "amdgcn-amd-amdpal-llvm-gfx802"
