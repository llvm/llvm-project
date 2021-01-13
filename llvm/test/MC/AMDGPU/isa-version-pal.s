// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx802 %s | FileCheck --check-prefix=OSABI-PAL %s
// RUN: llvm-mc -triple amdgcn-amd-amdpal -mcpu=iceland %s | FileCheck --check-prefix=OSABI-PAL %s
// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s

// OSABI-PAL: .amd_amdgpu_isa "amdgcn-amd-amdpal--gfx802"
// OSABI-UNK-ERR: error: .amd_amdgpu_isa directive does not match triple and/or mcpu arguments specified through the command line
// OSABI-HSA-ERR: error: .amd_amdgpu_isa directive does not match triple and/or mcpu arguments specified through the command line
// OSABI-PAL-ERR: error: .amd_amdgpu_isa directive does not match triple and/or mcpu arguments specified through the command line
.amd_amdgpu_isa "amdgcn-amd-amdpal--gfx802"
