// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding < %s | FileCheck -check-prefix=GFX13 %s

ds_direct_load v1 wait_va_vdst:15
// GFX13: ds_direct_load v1 wait_va_vdst:15 wait_vm_vsrc:0 ; encoding: [0x01,0x00,0x1f,0xce]

ds_direct_load v16 wait_va_vdst:0
// GFX13: ds_direct_load v16 wait_va_vdst:0 wait_vm_vsrc:0 ; encoding: [0x10,0x00,0x10,0xce]

ds_direct_load v17
// GFX13: ds_direct_load v17 wait_va_vdst:0 wait_vm_vsrc:0 ; encoding: [0x11,0x00,0x10,0xce]

ds_param_load v1, attr0.x wait_va_vdst:15
// GFX13: ds_param_load v1, attr0.x wait_va_vdst:15 wait_vm_vsrc:0 ; encoding: [0x01,0x00,0x0f,0xce]

ds_param_load v16, attr32.w wait_va_vdst:0
// GFX13: ds_param_load v16, attr32.w wait_va_vdst:0 wait_vm_vsrc:0 ; encoding: [0x10,0x83,0x00,0xce]

ds_param_load v17, attr32.w
// GFX13: ds_param_load v17, attr32.w wait_va_vdst:0 wait_vm_vsrc:0 ; encoding: [0x11,0x83,0x00,0xce]

ds_direct_load v1 wait_va_vdst:15 wait_vm_vsrc:1
// GFX13: ds_direct_load v1 wait_va_vdst:15 wait_vm_vsrc:1 ; encoding: [0x01,0x00,0x9f,0xce]

ds_direct_load v16 wait_va_vdst:0 wait_vm_vsrc:1
// GFX13: ds_direct_load v16 wait_va_vdst:0 wait_vm_vsrc:1 ; encoding: [0x10,0x00,0x90,0xce]

ds_direct_load v17 wait_vm_vsrc:1
// GFX13: ds_direct_load v17 wait_va_vdst:0 wait_vm_vsrc:1 ; encoding: [0x11,0x00,0x90,0xce]

ds_param_load v1, attr0.x wait_va_vdst:15 wait_vm_vsrc:1
// GFX13: ds_param_load v1, attr0.x wait_va_vdst:15 wait_vm_vsrc:1 ; encoding: [0x01,0x00,0x8f,0xce]

ds_param_load v16, attr32.w wait_va_vdst:0 wait_vm_vsrc:1
// GFX13: ds_param_load v16, attr32.w wait_va_vdst:0 wait_vm_vsrc:1 ; encoding: [0x10,0x83,0x80,0xce]

ds_param_load v17, attr32.w wait_vm_vsrc:1
// GFX13: ds_param_load v17, attr32.w wait_va_vdst:0 wait_vm_vsrc:1 ; encoding: [0x11,0x83,0x80,0xce]
