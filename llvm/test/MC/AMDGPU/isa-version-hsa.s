// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-unknown -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=4 -mcpu=gfx802 %s | FileCheck --check-prefix=OSABI-HSA %s
// RUN: llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=4 -mcpu=iceland %s | FileCheck --check-prefix=OSABI-HSA %s
// RUN: not llvm-mc -triple amdgcn-amd-amdhsa --amdhsa-code-object-version=4 -mcpu=gfx803 %s 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=OSABI-PAL-ERR %s
// RUN: not llvm-mc -triple amdgcn-amd-amdpal -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=OSABI-PAL-ERR %s

// OSABI-HSA: .amdgcn_target "amdgcn-amd-amdhsa--gfx802"
// OSABI-UNK-ERR: error: .amdgcn_target directive's target id amdgcn-amd-amdhsa--gfx802 does not match the specified target id amdgcn-amd-unknown--gfx802
// OSABI-HSA-ERR: error: .amdgcn_target directive's target id amdgcn-amd-amdhsa--gfx802 does not match the specified target id amdgcn-amd-amdhsa--gfx803
// OSABI-PAL-ERR: error: .amdgcn_target directive's target id amdgcn-amd-amdhsa--gfx802 does not match the specified target id amdgcn-amd-amdpal--gfx802
.amdgcn_target "amdgcn-amd-amdhsa--gfx802"
