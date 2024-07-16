// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=4 -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=4 -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx802 %s | FileCheck --check-prefix=OSABI-PAL %s
// RUN: llvm-mc -triple amdgcn-amd-amdpal -mcpu=iceland %s | FileCheck --check-prefix=OSABI-PAL %s
// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx803 %s 2>&1 | FileCheck --check-prefix=OSABI-PAL-ERR %s

// OSABI-PAL: .amd_amdgpu_isa "amdgcn-amd-amdpal--gfx802"
// OSABI-UNK-ERR: error: target id must match options
// OSABI-HSA-ERR: error: .amdgcn_target directive's target id amdgcn-amd-amdpal--gfx802 does not match the specified target id amdgcn-amd-amdhsa--gfx802
// OSABI-PAL-ERR: error: target id must match options
.amd_amdgpu_isa "amdgcn-amd-amdpal--gfx802"
.amdgcn_target "amdgcn-amd-amdpal--gfx802"
