// RUN: llvm-mc -triple=amdgpu9.42 -show-encoding %s | FileCheck --check-prefix=SUPPORTED %s
// RUN: llvm-mc -triple=amdgpu9.50 -show-encoding %s | FileCheck --check-prefix=SUPPORTED %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx9-4-generic -show-encoding %s | FileCheck --check-prefix=SUPPORTED %s
// RUN: not llvm-mc -triple=amdgpu9.0a -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX90A %s
// RUN: not llvm-mc -triple=amdgpu9.42 -mattr=-buffer-inv-inst -filetype=null %s 2>&1 | FileCheck --check-prefix=DISABLED %s

buffer_inv sc0 sc1
// SUPPORTED: buffer_inv sc0 sc1                    ; encoding: [0x00,0xc0,0xa4,0xe0,0x00,0x00,0x00,0x00]
// GFX90A: error: instruction not supported on this GPU (gfx90a): buffer_inv
// DISABLED: error: instruction not supported on this GPU (gfx942): buffer_inv
