// RUN: llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx802 %s | FileCheck --check-prefix=OSABI-UNK %s
// RUN: llvm-mc -triple amdgcn-amd-unknown -mcpu=iceland %s | FileCheck --check-prefix=OSABI-UNK %s
// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx803 %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=4 -mcpu=gfx802 %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=4 -mcpu=iceland %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx802 %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-PAL-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdpal -mcpu=iceland %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-PAL-ERR %s

// OSABI-UNK: .amd_amdgpu_isa "amdgcn-amd-unknown-unknown-gfx802"
// OSABI-UNK-ERR: error: .amd_amdgpu_isa directive's target id amdgcn-amd-unknown-unknown-gfx802 does not match the specified target id amdgcn-amd-unknown-unknown-gfx803
// OSABI-HSA-ERR: error: .amdgcn_target directive's target id amdgcn-amd-unknown-unknown-gfx802 does not match the specified target id amdgcn-amd-amdhsa-unknown-gfx802
// OSABI-PAL-ERR: error: .amdgcn_target directive's target id amdgcn-amd-unknown-unknown-gfx802 does not match the specified target id amdgcn-amd-amdpal-unknown-gfx802
.amd_amdgpu_isa "amdgcn-amd-unknown--gfx802"
.amdgcn_target "amdgcn-amd-unknown--gfx802"
