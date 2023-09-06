// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding < %s | FileCheck --check-prefix=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] ; encoding: [0x08,0x40,0x1f,0xcc,0x00,0x01,0x10,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] ; encoding: [0x08,0x40,0x1f,0xcc,0x00,0x01,0x10,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] ; encoding: [0x08,0x40,0x1f,0xcc,0x00,0x01,0x10,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] ; encoding: [0x08,0x40,0x1f,0xcc,0x00,0x01,0x10,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] op_sel_hi:[0,0,0]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] op_sel_hi:[0,0,0] ; encoding: [0x08,0x00,0x1f,0xcc,0x00,0x01,0x10,0x04]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] op_sel:[0,0,1] op_sel_hi:[0,0,1]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] op_sel:[0,0,1] op_sel_hi:[0,0,1] ; encoding: [0x08,0x60,0x1f,0xcc,0x00,0x01,0x10,0x04]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,1,1]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,1,1] ; encoding: [0x08,0x40,0x1f,0xcc,0x00,0x01,0x10,0xfc]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[1,1,1]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[1,1,1] ; encoding: [0x08,0x47,0x1f,0xcc,0x00,0x01,0x10,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,1,1] neg_hi:[1,1,1]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,1,1] neg_hi:[1,1,1] ; encoding: [0x08,0x47,0x1f,0xcc,0x00,0x01,0x10,0xfc]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,0,0]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,0,0] ; encoding: [0x08,0x40,0x1f,0xcc,0x00,0x01,0x10,0x3c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[0,1,0]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[0,1,0] ; encoding: [0x08,0x40,0x1f,0xcc,0x00,0x01,0x10,0x5c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[0,0,1]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[0,0,1] ; encoding: [0x08,0x40,0x1f,0xcc,0x00,0x01,0x10,0x9c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[1,0,0]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[1,0,0] ; encoding: [0x08,0x41,0x1f,0xcc,0x00,0x01,0x10,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[0,1,0]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[0,1,0] ; encoding: [0x08,0x42,0x1f,0xcc,0x00,0x01,0x10,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[0,0,1]
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[0,0,1] ; encoding: [0x08,0x44,0x1f,0xcc,0x00,0x01,0x10,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] clamp
// GFX1210: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] clamp ; encoding: [0x08,0xc0,0x1f,0xcc,0x00,0x01,0x10,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[0:1], v[4:5], v[8:9], v[16:17]
// GFX1210: v_pk_fma_f32 v[0:1], v[4:5], v[8:9], v[16:17] ; encoding: [0x00,0x40,0x1f,0xcc,0x04,0x11,0x42,0x1c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_fma_f32 v[0:1], v[2:3], v[4:5], 1.0
// GFX1210: v_pk_fma_f32 v[0:1], v[2:3], v[4:5], 1.0 ; encoding: [0x00,0x40,0x1f,0xcc,0x02,0x09,0xca,0x1b]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[254:255], v[8:9], v[16:17]
// GFX1210: v_pk_mul_f32 v[254:255], v[8:9], v[16:17] ; encoding: [0xfe,0x40,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[254:255], v[16:17]
// GFX1210: v_pk_mul_f32 v[4:5], v[254:255], v[16:17] ; encoding: [0x04,0x40,0x28,0xcc,0xfe,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], s[2:3], v[16:17]
// GFX1210: v_pk_mul_f32 v[4:5], s[2:3], v[16:17]   ; encoding: [0x04,0x40,0x28,0xcc,0x02,0x20,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], s[100:101], v[16:17]
// GFX1210: v_pk_mul_f32 v[4:5], s[100:101], v[16:17] ; encoding: [0x04,0x40,0x28,0xcc,0x64,0x20,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], vcc, v[16:17]
// GFX1210: v_pk_mul_f32 v[4:5], vcc, v[16:17]      ; encoding: [0x04,0x40,0x28,0xcc,0x6a,0x20,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], exec, v[16:17]
// GFX1210: v_pk_mul_f32 v[4:5], exec, v[16:17]     ; encoding: [0x04,0x40,0x28,0xcc,0x7e,0x20,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[254:255]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[254:255] ; encoding: [0x04,0x40,0x28,0xcc,0x08,0xfd,0x03,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], s[2:3]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], s[2:3]     ; encoding: [0x04,0x40,0x28,0xcc,0x08,0x05,0x00,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], s[100:101]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], s[100:101] ; encoding: [0x04,0x40,0x28,0xcc,0x08,0xc9,0x00,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], vcc
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], vcc        ; encoding: [0x04,0x40,0x28,0xcc,0x08,0xd5,0x00,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], exec
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], exec       ; encoding: [0x04,0x40,0x28,0xcc,0x08,0xfd,0x00,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17]   ; encoding: [0x04,0x40,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,0]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,0] ; encoding: [0x04,0x48,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[0,1]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[0,1] ; encoding: [0x04,0x50,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,1]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,1] ; encoding: [0x04,0x58,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17]   ; encoding: [0x04,0x40,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,0]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,0] ; encoding: [0x04,0x40,0x28,0xcc,0x08,0x21,0x02,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[1,0]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[1,0] ; encoding: [0x04,0x40,0x28,0xcc,0x08,0x21,0x02,0x08]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,1]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,1] ; encoding: [0x04,0x40,0x28,0xcc,0x08,0x21,0x02,0x10]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,0]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,0] ; encoding: [0x04,0x40,0x28,0xcc,0x08,0x21,0x02,0x38]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[0,1]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[0,1] ; encoding: [0x04,0x40,0x28,0xcc,0x08,0x21,0x02,0x58]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,1]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,1] ; encoding: [0x04,0x40,0x28,0xcc,0x08,0x21,0x02,0x78]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,0]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,0] ; encoding: [0x04,0x41,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[0,1]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[0,1] ; encoding: [0x04,0x42,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,1]
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,1] ; encoding: [0x04,0x43,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[4:5], v[8:9], v[16:17] clamp
// GFX1210: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] clamp ; encoding: [0x04,0xc0,0x28,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_mul_f32 v[0:1], v[2:3], 1.0
// GFX1210: v_pk_mul_f32 v[0:1], v[2:3], 1.0        ; encoding: [0x00,0x40,0x28,0xcc,0x02,0xe5,0x01,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[254:255], v[8:9], v[16:17]
// GFX1210: v_pk_add_f32 v[254:255], v[8:9], v[16:17] ; encoding: [0xfe,0x40,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[254:255], v[16:17]
// GFX1210: v_pk_add_f32 v[4:5], v[254:255], v[16:17] ; encoding: [0x04,0x40,0x29,0xcc,0xfe,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], s[2:3], v[16:17]
// GFX1210: v_pk_add_f32 v[4:5], s[2:3], v[16:17]   ; encoding: [0x04,0x40,0x29,0xcc,0x02,0x20,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], s[100:101], v[16:17]
// GFX1210: v_pk_add_f32 v[4:5], s[100:101], v[16:17] ; encoding: [0x04,0x40,0x29,0xcc,0x64,0x20,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], vcc, v[16:17]
// GFX1210: v_pk_add_f32 v[4:5], vcc, v[16:17]      ; encoding: [0x04,0x40,0x29,0xcc,0x6a,0x20,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], exec, v[16:17]
// GFX1210: v_pk_add_f32 v[4:5], exec, v[16:17]     ; encoding: [0x04,0x40,0x29,0xcc,0x7e,0x20,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[254:255]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[254:255] ; encoding: [0x04,0x40,0x29,0xcc,0x08,0xfd,0x03,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], s[2:3]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], s[2:3]     ; encoding: [0x04,0x40,0x29,0xcc,0x08,0x05,0x00,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], s[100:101]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], s[100:101] ; encoding: [0x04,0x40,0x29,0xcc,0x08,0xc9,0x00,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], vcc
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], vcc        ; encoding: [0x04,0x40,0x29,0xcc,0x08,0xd5,0x00,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], exec
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], exec       ; encoding: [0x04,0x40,0x29,0xcc,0x08,0xfd,0x00,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17]   ; encoding: [0x04,0x40,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,0]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,0] ; encoding: [0x04,0x48,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[0,1]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[0,1] ; encoding: [0x04,0x50,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,1]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,1] ; encoding: [0x04,0x58,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17]   ; encoding: [0x04,0x40,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,0]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,0] ; encoding: [0x04,0x40,0x29,0xcc,0x08,0x21,0x02,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[1,0]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[1,0] ; encoding: [0x04,0x40,0x29,0xcc,0x08,0x21,0x02,0x08]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,1]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,1] ; encoding: [0x04,0x40,0x29,0xcc,0x08,0x21,0x02,0x10]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,0]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,0] ; encoding: [0x04,0x40,0x29,0xcc,0x08,0x21,0x02,0x38]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[0,1]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[0,1] ; encoding: [0x04,0x40,0x29,0xcc,0x08,0x21,0x02,0x58]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,1]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,1] ; encoding: [0x04,0x40,0x29,0xcc,0x08,0x21,0x02,0x78]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,0]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,0] ; encoding: [0x04,0x41,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[0,1]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[0,1] ; encoding: [0x04,0x42,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,1]
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,1] ; encoding: [0x04,0x43,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[4:5], v[8:9], v[16:17] clamp
// GFX1210: v_pk_add_f32 v[4:5], v[8:9], v[16:17] clamp ; encoding: [0x04,0xc0,0x29,0xcc,0x08,0x21,0x02,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_pk_add_f32 v[0:1], v[2:3], 1.0
// GFX1210: v_pk_add_f32 v[0:1], v[2:3], 1.0        ; encoding: [0x00,0x40,0x29,0xcc,0x02,0xe5,0x01,0x18]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
