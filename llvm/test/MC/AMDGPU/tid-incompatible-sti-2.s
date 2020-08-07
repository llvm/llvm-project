// RUN: not llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdgcn-new-target-id %s 2>&1 | FileCheck %s

// CHECK: error: target must match options

.amdgcn_target "amdgcn-amd-amdhsa--gfx900:xnack+:sram-ecc+"
.text
