// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx800 %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=OSABI-UNK-ERR --check-prefix=GFX800 %s
// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=OSABI-UNK-ERR --check-prefix=GFX800 %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx800 %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=OSABI-HSA-ERR --check-prefix=GFX800 %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=OSABI-HSA-ERR --check-prefix=GFX800 %s
// RUN: llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx800 %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-PAL --check-prefix=GFX800 %s
// RUN: llvm-mc -triple amdgcn-amd-amdpal -mcpu=iceland %s | FileCheck --check-prefix=GCN --check-prefix=OSABI-PAL --check-prefix=GFX800 %s
// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx803 %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=OSABI-UNK-ERR --check-prefix=GFX800 %s

// OSABI-PAL: .amd_amdgpu_isa "amdgcn-amd-amdpal--gfx800"
// OSABI-UNK-ERR: error: .amd_amdgpu_isa directive does not match triple and/or mcpu arguments specified through the command line
// OSABI-HSA-ERR: error: .amd_amdgpu_isa directive does not match triple and/or mcpu arguments specified through the command line
// OSABI-PAL-ERR: error: .amd_amdgpu_isa directive does not match triple and/or mcpu arguments specified through the command line
.amd_amdgpu_isa "amdgcn-amd-amdpal--gfx800"
