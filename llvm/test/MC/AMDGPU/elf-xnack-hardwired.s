// Hardwired-on XNACK is implied by the processor, not an ELF mode selection.
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=4 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=449
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=5 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=449
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=449
// RUN: llvm-mc -triple=amdgpu12.50s-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=4EB
// RUN: llvm-mc -triple=amdgpu12.51-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=45A
// RUN: llvm-mc -triple=amdgpu12.5-amd-amdhsa --amdhsa-code-object-version=6 --amdgpu-force-generic-version=1 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s -DFLAGS=100045B

// CHECK: Flags [ (0x[[FLAGS]])
// CHECK: EF_AMDGPU_FEATURE_SRAMECC_ANY_V4

// SRAMECC remains selectable on gfx1250 even though XNACK is hardwired on.
// RUN: echo '.amdgcn_target "amdgpu12.50-amd-amdhsa--gfx1250:sramecc+"' | llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj | llvm-readobj --file-headers - | FileCheck %s --check-prefix=SRAMECC-ON
// RUN: echo '.amdgcn_target "amdgpu12.50-amd-amdhsa--gfx1250:sramecc-"' | llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj | llvm-readobj --file-headers - | FileCheck %s --check-prefix=SRAMECC-OFF

// SRAMECC-ON: Flags [ (0xC49)
// SRAMECC-ON: EF_AMDGPU_FEATURE_SRAMECC_ON_V4
// SRAMECC-OFF: Flags [ (0x849)
// SRAMECC-OFF: EF_AMDGPU_FEATURE_SRAMECC_OFF_V4

s_endpgm
