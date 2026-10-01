// Hardwired-on SRAMECC is implied by the processor, not an ELF mode selection.
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa --amdhsa-code-object-version=4 -mattr=-sramecc-on-off-modes -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s
// RUN: llvm-mc -triple=amdgpu9.08-amd-amdhsa --amdhsa-code-object-version=6 -mattr=-sramecc-on-off-modes -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s

// CHECK:      Flags [ (0x130)
// CHECK-NEXT:   EF_AMDGPU_FEATURE_XNACK_ANY_V4 (0x100)
// CHECK-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX908 (0x30)
// CHECK-NEXT: ]

.amdgcn_target "amdgcn-amd-amdhsa--gfx908:sramecc+"
