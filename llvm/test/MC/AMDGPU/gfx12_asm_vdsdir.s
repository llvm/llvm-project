// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s | FileCheck --strict-whitespace -check-prefix=GFX12 %s

ds_direct_load v1 wait_va_vdst:15
// GFX12: ds_direct_load v1 wait_va_vdst:15 wait_vm_vsrc:0 ; encoding: [0x01,0x00,0x1f,0xce]

ds_direct_load v2 wait_va_vdst:14
// GFX12: ds_direct_load v2 wait_va_vdst:14 wait_vm_vsrc:0 ; encoding: [0x02,0x00,0x1e,0xce]

ds_direct_load v3 wait_va_vdst:13
// GFX12: ds_direct_load v3 wait_va_vdst:13 wait_vm_vsrc:0 ; encoding: [0x03,0x00,0x1d,0xce]

ds_direct_load v4 wait_va_vdst:12
// GFX12: ds_direct_load v4 wait_va_vdst:12 wait_vm_vsrc:0 ; encoding: [0x04,0x00,0x1c,0xce]

ds_direct_load v5 wait_va_vdst:11
// GFX12: ds_direct_load v5 wait_va_vdst:11 wait_vm_vsrc:0 ; encoding: [0x05,0x00,0x1b,0xce]

ds_direct_load v6 wait_va_vdst:10
// GFX12: ds_direct_load v6 wait_va_vdst:10 wait_vm_vsrc:0 ; encoding: [0x06,0x00,0x1a,0xce]

ds_direct_load v7 wait_va_vdst:9
// GFX12: ds_direct_load v7 wait_va_vdst:9 wait_vm_vsrc:0 ; encoding: [0x07,0x00,0x19,0xce]

ds_direct_load v8 wait_va_vdst:8
// GFX12: ds_direct_load v8 wait_va_vdst:8 wait_vm_vsrc:0 ; encoding: [0x08,0x00,0x18,0xce]

ds_direct_load v9 wait_va_vdst:7
// GFX12: ds_direct_load v9 wait_va_vdst:7 wait_vm_vsrc:0 ; encoding: [0x09,0x00,0x17,0xce]

ds_direct_load v10 wait_va_vdst:6
// GFX12: ds_direct_load v10 wait_va_vdst:6 wait_vm_vsrc:0 ; encoding: [0x0a,0x00,0x16,0xce]

ds_direct_load v11 wait_va_vdst:5
// GFX12: ds_direct_load v11 wait_va_vdst:5 wait_vm_vsrc:0 ; encoding: [0x0b,0x00,0x15,0xce]

ds_direct_load v12 wait_va_vdst:4
// GFX12: ds_direct_load v12 wait_va_vdst:4 wait_vm_vsrc:0 ; encoding: [0x0c,0x00,0x14,0xce]

ds_direct_load v13 wait_va_vdst:3
// GFX12: ds_direct_load v13 wait_va_vdst:3 wait_vm_vsrc:0 ; encoding: [0x0d,0x00,0x13,0xce]

ds_direct_load v14 wait_va_vdst:2
// GFX12: ds_direct_load v14 wait_va_vdst:2 wait_vm_vsrc:0 ; encoding: [0x0e,0x00,0x12,0xce]

ds_direct_load v15 wait_va_vdst:1
// GFX12: ds_direct_load v15 wait_va_vdst:1 wait_vm_vsrc:0 ; encoding: [0x0f,0x00,0x11,0xce]

ds_direct_load v16 wait_va_vdst:0
// GFX12: ds_direct_load v16 wait_va_vdst:0 wait_vm_vsrc:0 ; encoding: [0x10,0x00,0x10,0xce]

ds_direct_load v17
// GFX12: ds_direct_load v17 wait_va_vdst:0 wait_vm_vsrc:0 ; encoding: [0x11,0x00,0x10,0xce]

ds_param_load v1, attr0.x wait_va_vdst:15
// GFX12: ds_param_load v1, attr0.x wait_va_vdst:15 wait_vm_vsrc:0 ; encoding: [0x01,0x00,0x0f,0xce]

ds_param_load v2, attr0.y wait_va_vdst:14
// GFX12: ds_param_load v2, attr0.y wait_va_vdst:14 wait_vm_vsrc:0 ; encoding: [0x02,0x01,0x0e,0xce]

ds_param_load v3, attr0.z wait_va_vdst:13
// GFX12: ds_param_load v3, attr0.z wait_va_vdst:13 wait_vm_vsrc:0 ; encoding: [0x03,0x02,0x0d,0xce]

ds_param_load v4, attr0.w wait_va_vdst:12
// GFX12: ds_param_load v4, attr0.w wait_va_vdst:12 wait_vm_vsrc:0 ; encoding: [0x04,0x03,0x0c,0xce]

ds_param_load v5, attr0.x wait_va_vdst:11
// GFX12: ds_param_load v5, attr0.x wait_va_vdst:11 wait_vm_vsrc:0 ; encoding: [0x05,0x00,0x0b,0xce]

ds_param_load v6, attr1.x wait_va_vdst:10
// GFX12: ds_param_load v6, attr1.x wait_va_vdst:10 wait_vm_vsrc:0 ; encoding: [0x06,0x04,0x0a,0xce]

ds_param_load v7, attr2.y wait_va_vdst:9
// GFX12: ds_param_load v7, attr2.y wait_va_vdst:9 wait_vm_vsrc:0 ; encoding: [0x07,0x09,0x09,0xce]

ds_param_load v8, attr3.z wait_va_vdst:8
// GFX12: ds_param_load v8, attr3.z wait_va_vdst:8 wait_vm_vsrc:0 ; encoding: [0x08,0x0e,0x08,0xce]

ds_param_load v9, attr4.w wait_va_vdst:7
// GFX12: ds_param_load v9, attr4.w wait_va_vdst:7 wait_vm_vsrc:0 ; encoding: [0x09,0x13,0x07,0xce]

ds_param_load v10, attr11.x wait_va_vdst:6
// GFX12: ds_param_load v10, attr11.x wait_va_vdst:6 wait_vm_vsrc:0 ; encoding: [0x0a,0x2c,0x06,0xce]

ds_param_load v11, attr22.y wait_va_vdst:5
// GFX12: ds_param_load v11, attr22.y wait_va_vdst:5 wait_vm_vsrc:0 ; encoding: [0x0b,0x59,0x05,0xce]

ds_param_load v13, attr32.x wait_va_vdst:3
// GFX12: ds_param_load v13, attr32.x wait_va_vdst:3 wait_vm_vsrc:0 ; encoding: [0x0d,0x80,0x03,0xce]

ds_param_load v14, attr32.y wait_va_vdst:2
// GFX12: ds_param_load v14, attr32.y wait_va_vdst:2 wait_vm_vsrc:0 ; encoding: [0x0e,0x81,0x02,0xce]

ds_param_load v15, attr32.z wait_va_vdst:1
// GFX12: ds_param_load v15, attr32.z wait_va_vdst:1 wait_vm_vsrc:0 ; encoding: [0x0f,0x82,0x01,0xce]

ds_param_load v16, attr32.w wait_va_vdst:0
// GFX12: ds_param_load v16, attr32.w wait_va_vdst:0 wait_vm_vsrc:0 ; encoding: [0x10,0x83,0x00,0xce]

ds_param_load v17, attr32.w
// GFX12: ds_param_load v17, attr32.w wait_va_vdst:0 wait_vm_vsrc:0 ; encoding: [0x11,0x83,0x00,0xce]

ds_direct_load v1 wait_va_vdst:15 wait_vm_vsrc:1
// GFX12: ds_direct_load v1 wait_va_vdst:15 wait_vm_vsrc:1 ; encoding: [0x01,0x00,0x9f,0xce]

ds_direct_load v2 wait_va_vdst:14 wait_vm_vsrc:1
// GFX12: ds_direct_load v2 wait_va_vdst:14 wait_vm_vsrc:1 ; encoding: [0x02,0x00,0x9e,0xce]

ds_direct_load v3 wait_va_vdst:13 wait_vm_vsrc:1
// GFX12: ds_direct_load v3 wait_va_vdst:13 wait_vm_vsrc:1 ; encoding: [0x03,0x00,0x9d,0xce]

ds_direct_load v4 wait_va_vdst:12 wait_vm_vsrc:1
// GFX12: ds_direct_load v4 wait_va_vdst:12 wait_vm_vsrc:1 ; encoding: [0x04,0x00,0x9c,0xce]

ds_direct_load v5 wait_va_vdst:11 wait_vm_vsrc:1
// GFX12: ds_direct_load v5 wait_va_vdst:11 wait_vm_vsrc:1 ; encoding: [0x05,0x00,0x9b,0xce]

ds_direct_load v6 wait_va_vdst:10 wait_vm_vsrc:1
// GFX12: ds_direct_load v6 wait_va_vdst:10 wait_vm_vsrc:1 ; encoding: [0x06,0x00,0x9a,0xce]

ds_direct_load v7 wait_va_vdst:9 wait_vm_vsrc:1
// GFX12: ds_direct_load v7 wait_va_vdst:9 wait_vm_vsrc:1 ; encoding: [0x07,0x00,0x99,0xce]

ds_direct_load v8 wait_va_vdst:8 wait_vm_vsrc:1
// GFX12: ds_direct_load v8 wait_va_vdst:8 wait_vm_vsrc:1 ; encoding: [0x08,0x00,0x98,0xce]

ds_direct_load v9 wait_va_vdst:7 wait_vm_vsrc:1
// GFX12: ds_direct_load v9 wait_va_vdst:7 wait_vm_vsrc:1 ; encoding: [0x09,0x00,0x97,0xce]

ds_direct_load v10 wait_va_vdst:6 wait_vm_vsrc:1
// GFX12: ds_direct_load v10 wait_va_vdst:6 wait_vm_vsrc:1 ; encoding: [0x0a,0x00,0x96,0xce]

ds_direct_load v11 wait_va_vdst:5 wait_vm_vsrc:1
// GFX12: ds_direct_load v11 wait_va_vdst:5 wait_vm_vsrc:1 ; encoding: [0x0b,0x00,0x95,0xce]

ds_direct_load v12 wait_va_vdst:4 wait_vm_vsrc:1
// GFX12: ds_direct_load v12 wait_va_vdst:4 wait_vm_vsrc:1 ; encoding: [0x0c,0x00,0x94,0xce]

ds_direct_load v13 wait_va_vdst:3 wait_vm_vsrc:1
// GFX12: ds_direct_load v13 wait_va_vdst:3 wait_vm_vsrc:1 ; encoding: [0x0d,0x00,0x93,0xce]

ds_direct_load v14 wait_va_vdst:2 wait_vm_vsrc:1
// GFX12: ds_direct_load v14 wait_va_vdst:2 wait_vm_vsrc:1 ; encoding: [0x0e,0x00,0x92,0xce]

ds_direct_load v15 wait_va_vdst:1 wait_vm_vsrc:1
// GFX12: ds_direct_load v15 wait_va_vdst:1 wait_vm_vsrc:1 ; encoding: [0x0f,0x00,0x91,0xce]

ds_direct_load v16 wait_va_vdst:0 wait_vm_vsrc:1
// GFX12: ds_direct_load v16 wait_va_vdst:0 wait_vm_vsrc:1 ; encoding: [0x10,0x00,0x90,0xce]

ds_direct_load v17 wait_vm_vsrc:1
// GFX12: ds_direct_load v17 wait_va_vdst:0 wait_vm_vsrc:1 ; encoding: [0x11,0x00,0x90,0xce]

ds_param_load v1, attr0.x wait_va_vdst:15 wait_vm_vsrc:1
// GFX12: ds_param_load v1, attr0.x wait_va_vdst:15 wait_vm_vsrc:1 ; encoding: [0x01,0x00,0x8f,0xce]

ds_param_load v2, attr0.y wait_va_vdst:14 wait_vm_vsrc:1
// GFX12: ds_param_load v2, attr0.y wait_va_vdst:14 wait_vm_vsrc:1 ; encoding: [0x02,0x01,0x8e,0xce]

ds_param_load v3, attr0.z wait_va_vdst:13 wait_vm_vsrc:1
// GFX12: ds_param_load v3, attr0.z wait_va_vdst:13 wait_vm_vsrc:1 ; encoding: [0x03,0x02,0x8d,0xce]

ds_param_load v4, attr0.w wait_va_vdst:12 wait_vm_vsrc:1
// GFX12: ds_param_load v4, attr0.w wait_va_vdst:12 wait_vm_vsrc:1 ; encoding: [0x04,0x03,0x8c,0xce]

ds_param_load v5, attr0.x wait_va_vdst:11 wait_vm_vsrc:1
// GFX12: ds_param_load v5, attr0.x wait_va_vdst:11 wait_vm_vsrc:1 ; encoding: [0x05,0x00,0x8b,0xce]

ds_param_load v6, attr1.x wait_va_vdst:10 wait_vm_vsrc:1
// GFX12: ds_param_load v6, attr1.x wait_va_vdst:10 wait_vm_vsrc:1 ; encoding: [0x06,0x04,0x8a,0xce]

ds_param_load v7, attr2.y wait_va_vdst:9 wait_vm_vsrc:1
// GFX12: ds_param_load v7, attr2.y wait_va_vdst:9 wait_vm_vsrc:1 ; encoding: [0x07,0x09,0x89,0xce]

ds_param_load v8, attr3.z wait_va_vdst:8 wait_vm_vsrc:1
// GFX12: ds_param_load v8, attr3.z wait_va_vdst:8 wait_vm_vsrc:1 ; encoding: [0x08,0x0e,0x88,0xce]

ds_param_load v9, attr4.w wait_va_vdst:7 wait_vm_vsrc:1
// GFX12: ds_param_load v9, attr4.w wait_va_vdst:7 wait_vm_vsrc:1 ; encoding: [0x09,0x13,0x87,0xce]

ds_param_load v10, attr11.x wait_va_vdst:6 wait_vm_vsrc:1
// GFX12: ds_param_load v10, attr11.x wait_va_vdst:6 wait_vm_vsrc:1 ; encoding: [0x0a,0x2c,0x86,0xce]

ds_param_load v11, attr22.y wait_va_vdst:5 wait_vm_vsrc:1
// GFX12: ds_param_load v11, attr22.y wait_va_vdst:5 wait_vm_vsrc:1 ; encoding: [0x0b,0x59,0x85,0xce]

ds_param_load v13, attr32.x wait_va_vdst:3 wait_vm_vsrc:1
// GFX12: ds_param_load v13, attr32.x wait_va_vdst:3 wait_vm_vsrc:1 ; encoding: [0x0d,0x80,0x83,0xce]

ds_param_load v14, attr32.y wait_va_vdst:2 wait_vm_vsrc:1
// GFX12: ds_param_load v14, attr32.y wait_va_vdst:2 wait_vm_vsrc:1 ; encoding: [0x0e,0x81,0x82,0xce]

ds_param_load v15, attr32.z wait_va_vdst:1 wait_vm_vsrc:1
// GFX12: ds_param_load v15, attr32.z wait_va_vdst:1 wait_vm_vsrc:1 ; encoding: [0x0f,0x82,0x81,0xce]

ds_param_load v16, attr32.w wait_va_vdst:0 wait_vm_vsrc:1
// GFX12: ds_param_load v16, attr32.w wait_va_vdst:0 wait_vm_vsrc:1 ; encoding: [0x10,0x83,0x80,0xce]

ds_param_load v17, attr32.w wait_vm_vsrc:1
// GFX12: ds_param_load v17, attr32.w wait_va_vdst:0 wait_vm_vsrc:1 ; encoding: [0x11,0x83,0x80,0xce]
