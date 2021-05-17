// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -mcpu=gfx802 %s | FileCheck --check-prefix=OSABI-HSA %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -mcpu=iceland %s | FileCheck --check-prefix=OSABI-HSA %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=2 -mcpu=gfx803 %s 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-PAL-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdpal -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=OSABI-PAL-ERR %s

// OSABI-HSA: .amd_amdgpu_isa "amdgcn-amd-amdhsa--gfx802"
// OSABI-UNK-ERR: error: target id must match options
// OSABI-HSA-ERR: error: target id must match options
// OSABI-PAL-ERR: error: target id must match options
.amd_amdgpu_isa "amdgcn-amd-amdhsa--gfx802"
