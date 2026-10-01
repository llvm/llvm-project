// Code object V4/V5 retain XNACK_ON; V6 infers hardwired-on XNACK from the processor.
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=5 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s --check-prefix=V5
// RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj %s | llvm-readobj --file-headers - | FileCheck %s --check-prefix=V6

// V5: Flags [ (0x749)
// V6: Flags [ (0x449)

s_endpgm
