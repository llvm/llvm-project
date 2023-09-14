// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck -check-prefix=GFX11 %s

lds_direct_load v1 wait_vdst:15
// GFX11: lds_direct_load v1 wait_vdst:15  ; encoding: [0x01,0x00,0x1f,0xce]

lds_direct_load v2 wait_vdst:14
// GFX11: lds_direct_load v2 wait_vdst:14  ; encoding: [0x02,0x00,0x1e,0xce]

lds_direct_load v3 wait_vdst:13
// GFX11: lds_direct_load v3 wait_vdst:13  ; encoding: [0x03,0x00,0x1d,0xce]

lds_direct_load v4 wait_vdst:12
// GFX11: lds_direct_load v4 wait_vdst:12  ; encoding: [0x04,0x00,0x1c,0xce]

lds_direct_load v5 wait_vdst:11
// GFX11: lds_direct_load v5 wait_vdst:11  ; encoding: [0x05,0x00,0x1b,0xce]

lds_direct_load v6 wait_vdst:10
// GFX11: lds_direct_load v6 wait_vdst:10  ; encoding: [0x06,0x00,0x1a,0xce]

lds_direct_load v7 wait_vdst:9
// GFX11: lds_direct_load v7 wait_vdst:9   ; encoding: [0x07,0x00,0x19,0xce]

lds_direct_load v8 wait_vdst:8
// GFX11: lds_direct_load v8 wait_vdst:8   ; encoding: [0x08,0x00,0x18,0xce]

lds_direct_load v9 wait_vdst:7
// GFX11: lds_direct_load v9 wait_vdst:7   ; encoding: [0x09,0x00,0x17,0xce]

lds_direct_load v10 wait_vdst:6
// GFX11: lds_direct_load v10 wait_vdst:6  ; encoding: [0x0a,0x00,0x16,0xce]

lds_direct_load v11 wait_vdst:5
// GFX11: lds_direct_load v11 wait_vdst:5  ; encoding: [0x0b,0x00,0x15,0xce]

lds_direct_load v12 wait_vdst:4
// GFX11: lds_direct_load v12 wait_vdst:4  ; encoding: [0x0c,0x00,0x14,0xce]

lds_direct_load v13 wait_vdst:3
// GFX11: lds_direct_load v13 wait_vdst:3  ; encoding: [0x0d,0x00,0x13,0xce]

lds_direct_load v14 wait_vdst:2
// GFX11: lds_direct_load v14 wait_vdst:2  ; encoding: [0x0e,0x00,0x12,0xce]

lds_direct_load v15 wait_vdst:1
// GFX11: lds_direct_load v15 wait_vdst:1  ; encoding: [0x0f,0x00,0x11,0xce]

lds_direct_load v16 wait_vdst:0
// GFX11: lds_direct_load v16  ; encoding: [0x10,0x00,0x10,0xce]

lds_direct_load v17
// GFX11: lds_direct_load v17  ; encoding: [0x11,0x00,0x10,0xce]

lds_param_load v1, attr0.x wait_vdst:15
// GFX11: lds_param_load v1, attr0.x wait_vdst:15   ; encoding: [0x01,0x00,0x0f,0xce]

lds_param_load v2, attr0.y wait_vdst:14
// GFX11: lds_param_load v2, attr0.y wait_vdst:14   ; encoding: [0x02,0x01,0x0e,0xce]

lds_param_load v3, attr0.z wait_vdst:13
// GFX11: lds_param_load v3, attr0.z wait_vdst:13   ; encoding: [0x03,0x02,0x0d,0xce]

lds_param_load v4, attr0.w wait_vdst:12
// GFX11: lds_param_load v4, attr0.w wait_vdst:12   ; encoding: [0x04,0x03,0x0c,0xce]

lds_param_load v5, attr0.x wait_vdst:11
// GFX11: lds_param_load v5, attr0.x wait_vdst:11   ; encoding: [0x05,0x00,0x0b,0xce]

lds_param_load v6, attr1.x wait_vdst:10
// GFX11: lds_param_load v6, attr1.x wait_vdst:10   ; encoding: [0x06,0x04,0x0a,0xce]

lds_param_load v7, attr2.y wait_vdst:9
// GFX11: lds_param_load v7, attr2.y wait_vdst:9    ; encoding: [0x07,0x09,0x09,0xce]

lds_param_load v8, attr3.z wait_vdst:8
// GFX11: lds_param_load v8, attr3.z wait_vdst:8    ; encoding: [0x08,0x0e,0x08,0xce]

lds_param_load v9, attr4.w wait_vdst:7
// GFX11: lds_param_load v9, attr4.w wait_vdst:7    ; encoding: [0x09,0x13,0x07,0xce]

lds_param_load v10, attr11.x wait_vdst:6
// GFX11: lds_param_load v10, attr11.x wait_vdst:6  ; encoding: [0x0a,0x2c,0x06,0xce]

lds_param_load v11, attr22.y wait_vdst:5
// GFX11: lds_param_load v11, attr22.y wait_vdst:5  ; encoding: [0x0b,0x59,0x05,0xce]

lds_param_load v12, attr32.z wait_vdst:4
// GFX11: lds_param_load v12, attr32.z wait_vdst:4  ; encoding: [0x0c,0x82,0x04,0xce]
