// RUN: llvm-mc -triple=amdgpu8.02-amd-unknown %s | FileCheck --check-prefix=OSABI-UNK %s
// RUN: llvm-mc -triple=amdgpu8.02-amd-unknown %s | FileCheck --check-prefix=OSABI-UNK %s
// RUN: not llvm-mc -triple=amdgpu8.03-amd-unknown %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-UNK-ERR %s
// RUN: not llvm-mc -triple=amdgpu8.02-amd-amdhsa --amdhsa-code-object-version=4 %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: not llvm-mc -triple=amdgpu8.02-amd-amdhsa --amdhsa-code-object-version=4 %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-HSA-ERR %s
// RUN: not llvm-mc -triple=amdgpu8.02-amd-amdpal %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-PAL-ERR %s
// RUN: not llvm-mc -triple=amdgpu8.02-amd-amdpal %s -filetype=null 2>&1 | FileCheck --check-prefix=OSABI-PAL-ERR %s

// OSABI-UNK: .amd_amdgpu_isa "amdgpu8.02-amd-unknown-unknown-gfx802"
// OSABI-UNK-ERR: error: target id 'amdgcn-amd-unknown--gfx802' specifies a processor that is not valid for subarch 'amdgpu8.03'
// OSABI-HSA-ERR: error: .amdgcn_target amdgcn-amd-unknown-unknown-gfx802 is incompatible with amdgpu8.02-amd-amdhsa-unknown-gfx802
// OSABI-PAL-ERR: error: .amdgcn_target amdgcn-amd-unknown-unknown-gfx802 is incompatible with amdgpu8.02-amd-amdpal-unknown-gfx802
.amd_amdgpu_isa "amdgcn-amd-unknown--gfx802"
.amdgcn_target "amdgcn-amd-unknown--gfx802"
