// RUN: sed 's/COV/4/g' %s | llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj | \
// RUN:   llvm-readobj --file-headers - | FileCheck %s --check-prefixes=HS4

// RUN: sed 's/COV/5/g' %s | llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj | \
// RUN:   llvm-readobj --file-headers - | FileCheck %s --check-prefixes=HS5

// RUN: sed 's/COV/4/g' %s | not llvm-mc -triple amdgcn-amd-amdpal -mcpu=gfx802 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR

// RUN: sed 's/COV/4/g' %s | not llvm-mc -triple amdgcn-amd-mesa3d -mcpu=gfx802 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR

// RUN: sed 's/COV/4/g' %s | not llvm-mc -triple amdgcn-amd- -mcpu=gfx802 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR

.amdhsa_code_object_version COV

// ERR: error: unknown directive

// HS4: OS/ABI: AMDGPU_HSA (0x40)
// HS4-NEXT: ABIVersion: 2

// HS5: OS/ABI: AMDGPU_HSA (0x40)
// HS5-NEXT: ABIVersion: 3
