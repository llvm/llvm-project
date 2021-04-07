// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck --check-prefixes=GFX908,NOT-GFX90A --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=GFX1010,NOT-GFX90A --implicit-check-not=error: %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx90a -show-encoding %s | FileCheck --check-prefix=GFX90A %s

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] ; encoding: [0x08,0x40,0xb0,0xd3,0x00,0x01,0x10,0x1c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] ; encoding: [0x08,0x40,0xb0,0xd3,0x00,0x01,0x10,0x1c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] ; encoding: [0x08,0x40,0xb0,0xd3,0x00,0x01,0x10,0x1c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] ; encoding: [0x08,0x40,0xb0,0xd3,0x00,0x01,0x10,0x1c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] op_sel_hi:[0,0,0] ; encoding: [0x08,0x00,0xb0,0xd3,0x00,0x01,0x10,0x04]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] op_sel_hi:[0,0,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] op_sel:[0,0,1] op_sel_hi:[0,0,1] ; encoding: [0x08,0x60,0xb0,0xd3,0x00,0x01,0x10,0x04]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] op_sel:[0,0,1] op_sel_hi:[0,0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,1,1] ; encoding: [0x08,0x40,0xb0,0xd3,0x00,0x01,0x10,0xfc]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[1,1,1] ; encoding: [0x08,0x47,0xb0,0xd3,0x00,0x01,0x10,0x1c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[1,1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,1,1] neg_hi:[1,1,1] ; encoding: [0x08,0x47,0xb0,0xd3,0x00,0x01,0x10,0xfc]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,1,1] neg_hi:[1,1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,0,0] ; encoding: [0x08,0x40,0xb0,0xd3,0x00,0x01,0x10,0x3c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[1,0,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[0,1,0] ; encoding: [0x08,0x40,0xb0,0xd3,0x00,0x01,0x10,0x5c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[0,1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[0,0,1] ; encoding: [0x08,0x40,0xb0,0xd3,0x00,0x01,0x10,0x9c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_lo:[0,0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[1,0,0] ; encoding: [0x08,0x41,0xb0,0xd3,0x00,0x01,0x10,0x1c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[1,0,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[0,1,0] ; encoding: [0x08,0x42,0xb0,0xd3,0x00,0x01,0x10,0x1c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[0,1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[0,0,1] ; encoding: [0x08,0x44,0xb0,0xd3,0x00,0x01,0x10,0x1c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] neg_hi:[0,0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] clamp ; encoding: [0x08,0xc0,0xb0,0xd3,0x00,0x01,0x10,0x1c]
v_pk_fma_f32 v[8:9], v[0:1], s[0:1], v[4:5] clamp

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_fma_f32 v[0:1], v[4:5], v[8:9], v[16:17] ; encoding: [0x00,0x40,0xb0,0xd3,0x04,0x11,0x42,0x1c]
v_pk_fma_f32 v[0:1], v[4:5], v[8:9], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[254:255], v[8:9], v[16:17] ; encoding: [0xfe,0x40,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[254:255], v[8:9], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[254:255], v[16:17] ; encoding: [0x04,0x40,0xb1,0xd3,0xfe,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[254:255], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], s[2:3], v[16:17] ; encoding: [0x04,0x40,0xb1,0xd3,0x02,0x20,0x02,0x18]
v_pk_mul_f32 v[4:5], s[2:3], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], s[100:101], v[16:17] ; encoding: [0x04,0x40,0xb1,0xd3,0x64,0x20,0x02,0x18]
v_pk_mul_f32 v[4:5], s[100:101], v[16:17]

// GFX90A: v_pk_mul_f32 v[4:5], flat_scratch, v[16:17] ; encoding: [0x04,0x40,0xb1,0xd3,0x66,0x20,0x02,0x18]
// NOT-GFX90A: error: instruction not supported on this GPU
v_pk_mul_f32 v[4:5], flat_scratch, v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], vcc, v[16:17] ; encoding: [0x04,0x40,0xb1,0xd3,0x6a,0x20,0x02,0x18]
v_pk_mul_f32 v[4:5], vcc, v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], exec, v[16:17] ; encoding: [0x04,0x40,0xb1,0xd3,0x7e,0x20,0x02,0x18]
v_pk_mul_f32 v[4:5], exec, v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[254:255] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0xfd,0x03,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[254:255]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], s[2:3] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0x05,0x00,0x18]
v_pk_mul_f32 v[4:5], v[8:9], s[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], s[100:101] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0xc9,0x00,0x18]
v_pk_mul_f32 v[4:5], v[8:9], s[100:101]

// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], flat_scratch ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0xcd,0x00,0x18]
// NOT-GFX90A: error: instruction not supported on this GPU
v_pk_mul_f32 v[4:5], v[8:9], flat_scratch

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], vcc ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0xd5,0x00,0x18]
v_pk_mul_f32 v[4:5], v[8:9], vcc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], exec ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0xfd,0x00,0x18]
v_pk_mul_f32 v[4:5], v[8:9], exec

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,0] ; encoding: [0x04,0x48,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[0,1] ; encoding: [0x04,0x50,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,1] ; encoding: [0x04,0x58,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,0] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0x21,0x02,0x00]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[1,0] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0x21,0x02,0x08]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,1] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0x21,0x02,0x10]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,0] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0x21,0x02,0x38]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[0,1] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0x21,0x02,0x58]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,1] ; encoding: [0x04,0x40,0xb1,0xd3,0x08,0x21,0x02,0x78]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,0] ; encoding: [0x04,0x41,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[0,1] ; encoding: [0x04,0x42,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,1] ; encoding: [0x04,0x43,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mul_f32 v[4:5], v[8:9], v[16:17] clamp ; encoding: [0x04,0xc0,0xb1,0xd3,0x08,0x21,0x02,0x18]
v_pk_mul_f32 v[4:5], v[8:9], v[16:17] clamp

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[254:255], v[8:9], v[16:17] ; encoding: [0xfe,0x40,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[254:255], v[8:9], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[254:255], v[16:17] ; encoding: [0x04,0x40,0xb2,0xd3,0xfe,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[254:255], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], s[2:3], v[16:17] ; encoding: [0x04,0x40,0xb2,0xd3,0x02,0x20,0x02,0x18]
v_pk_add_f32 v[4:5], s[2:3], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], s[100:101], v[16:17] ; encoding: [0x04,0x40,0xb2,0xd3,0x64,0x20,0x02,0x18]
v_pk_add_f32 v[4:5], s[100:101], v[16:17]

// GFX90A: v_pk_add_f32 v[4:5], flat_scratch, v[16:17] ; encoding: [0x04,0x40,0xb2,0xd3,0x66,0x20,0x02,0x18]
// NOT-GFX90A: error: instruction not supported on this GPU
v_pk_add_f32 v[4:5], flat_scratch, v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], vcc, v[16:17] ; encoding: [0x04,0x40,0xb2,0xd3,0x6a,0x20,0x02,0x18]
v_pk_add_f32 v[4:5], vcc, v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], exec, v[16:17] ; encoding: [0x04,0x40,0xb2,0xd3,0x7e,0x20,0x02,0x18]
v_pk_add_f32 v[4:5], exec, v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[254:255] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0xfd,0x03,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[254:255]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], s[2:3] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0x05,0x00,0x18]
v_pk_add_f32 v[4:5], v[8:9], s[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], s[100:101] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0xc9,0x00,0x18]
v_pk_add_f32 v[4:5], v[8:9], s[100:101]

// GFX90A: v_pk_add_f32 v[4:5], v[8:9], flat_scratch ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0xcd,0x00,0x18]
// NOT-GFX90A: error: instruction not supported on this GPU
v_pk_add_f32 v[4:5], v[8:9], flat_scratch

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], vcc ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0xd5,0x00,0x18]
v_pk_add_f32 v[4:5], v[8:9], vcc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], exec ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0xfd,0x00,0x18]
v_pk_add_f32 v[4:5], v[8:9], exec

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,0] ; encoding: [0x04,0x48,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[0,1] ; encoding: [0x04,0x50,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,1] ; encoding: [0x04,0x58,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel:[1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[16:17]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,0] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0x21,0x02,0x00]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[1,0] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0x21,0x02,0x08]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,1] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0x21,0x02,0x10]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] op_sel_hi:[0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,0] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0x21,0x02,0x38]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[0,1] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0x21,0x02,0x58]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,1] ; encoding: [0x04,0x40,0xb2,0xd3,0x08,0x21,0x02,0x78]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_lo:[1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,0] ; encoding: [0x04,0x41,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[0,1] ; encoding: [0x04,0x42,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,1] ; encoding: [0x04,0x43,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] neg_hi:[1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_add_f32 v[4:5], v[8:9], v[16:17] clamp ; encoding: [0x04,0xc0,0xb2,0xd3,0x08,0x21,0x02,0x18]
v_pk_add_f32 v[4:5], v[8:9], v[16:17] clamp

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mov_b32 v[0:1], v[2:3], v[4:5] ; encoding: [0x00,0x40,0xb3,0xd3,0x02,0x09,0x02,0x18]
v_pk_mov_b32 v[0:1], v[2:3], v[4:5]

// GFX90A: v_pk_mov_b32 v[0:1], flat_scratch, v[4:5] ; encoding: [0x00,0x40,0xb3,0xd3,0x66,0x08,0x02,0x18]
// NOT-GFX90A: error: instruction not supported on this GPU
v_pk_mov_b32 v[0:1], flat_scratch, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mov_b32 v[0:1], v[2:3], vcc ; encoding: [0x00,0x40,0xb3,0xd3,0x02,0xd5,0x00,0x18]
v_pk_mov_b32 v[0:1], v[2:3], vcc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mov_b32 v[0:1], v[2:3], s[0:1] ; encoding: [0x00,0x40,0xb3,0xd3,0x02,0x01,0x00,0x18]
v_pk_mov_b32 v[0:1], v[2:3], s[0:1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mov_b32 v[0:1], v[2:3], v[2:3] op_sel_hi:[0,1] ; encoding: [0x00,0x40,0xb3,0xd3,0x02,0x05,0x02,0x10]
v_pk_mov_b32 v[0:1], v[2:3], v[2:3] op_sel_hi:[0,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mov_b32 v[0:1], v[2:3], v[4:5] op_sel:[1,0] ; encoding: [0x00,0x48,0xb3,0xd3,0x02,0x09,0x02,0x18]
v_pk_mov_b32 v[0:1], v[2:3], v[4:5] op_sel:[1,0]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_pk_mov_b32 v[0:1], v[2:3], v[4:5] op_sel:[1,1] ; encoding: [0x00,0x58,0xb3,0xd3,0x02,0x09,0x02,0x18]
v_pk_mov_b32 v[0:1], v[2:3], v[4:5] op_sel:[1,1]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_wbl2 ; encoding: [0x00,0x00,0xa0,0xe0,0x00,0x00,0x00,0x00]
buffer_wbl2

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_invl2 ; encoding: [0x00,0x00,0xa4,0xe0,0x00,0x00,0x00,0x00]
buffer_invl2

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe1,0x00,0x04,0x03,0x03]
buffer_atomic_add_f64 v[4:5], off, s[12:15], s3 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe1,0x00,0x04,0x18,0x03]
buffer_atomic_add_f64 v[4:5], off, s[96:99], s3 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe1,0x00,0x04,0x02,0x65]
buffer_atomic_add_f64 v[4:5], off, s[8:11], s101 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe1,0x00,0x04,0x02,0x7c]
buffer_atomic_add_f64 v[4:5], off, s[8:11], m0 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe1,0x00,0x04,0x02,0x80]
buffer_atomic_add_f64 v[4:5], off, s[8:11], 0 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe1,0x00,0x04,0x02,0xc1]
buffer_atomic_add_f64 v[4:5], off, s[8:11], -1 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x3c,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_add_f64 v[4:5], v0, s[8:11], s3 idxen offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x3c,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_add_f64 v[4:5], v0, s[8:11], s3 offen offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 ; encoding: [0x00,0x00,0x3c,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_add_f64 v[4:5], off, s[8:11], s3

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 ; encoding: [0x00,0x00,0x3c,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_add_f64 v[4:5], off, s[8:11], s3

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x3c,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 offset:7

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x3e,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_add_f64 v[4:5], off, s[8:11], s3 offset:4095 slc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe1,0x00,0x04,0x03,0x03]
buffer_atomic_min_f64 v[4:5], off, s[12:15], s3 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe1,0x00,0x04,0x18,0x03]
buffer_atomic_min_f64 v[4:5], off, s[96:99], s3 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe1,0x00,0x04,0x02,0x65]
buffer_atomic_min_f64 v[4:5], off, s[8:11], s101 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe1,0x00,0x04,0x02,0x7c]
buffer_atomic_min_f64 v[4:5], off, s[8:11], m0 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe1,0x00,0x04,0x02,0x80]
buffer_atomic_min_f64 v[4:5], off, s[8:11], 0 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe1,0x00,0x04,0x02,0xc1]
buffer_atomic_min_f64 v[4:5], off, s[8:11], -1 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x40,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_min_f64 v[4:5], v0, s[8:11], s3 idxen offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x40,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_min_f64 v[4:5], v0, s[8:11], s3 offen offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 ; encoding: [0x00,0x00,0x40,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_min_f64 v[4:5], off, s[8:11], s3

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 ; encoding: [0x00,0x00,0x40,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_min_f64 v[4:5], off, s[8:11], s3

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x40,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 offset:7

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x42,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_min_f64 v[4:5], off, s[8:11], s3 offset:4095 slc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe1,0x00,0x04,0x03,0x03]
buffer_atomic_max_f64 v[4:5], off, s[12:15], s3 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe1,0x00,0x04,0x18,0x03]
buffer_atomic_max_f64 v[4:5], off, s[96:99], s3 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe1,0x00,0x04,0x02,0x65]
buffer_atomic_max_f64 v[4:5], off, s[8:11], s101 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe1,0x00,0x04,0x02,0x7c]
buffer_atomic_max_f64 v[4:5], off, s[8:11], m0 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe1,0x00,0x04,0x02,0x80]
buffer_atomic_max_f64 v[4:5], off, s[8:11], 0 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe1,0x00,0x04,0x02,0xc1]
buffer_atomic_max_f64 v[4:5], off, s[8:11], -1 offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x44,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_max_f64 v[4:5], v0, s[8:11], s3 idxen offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x44,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_max_f64 v[4:5], v0, s[8:11], s3 offen offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 ; encoding: [0x00,0x00,0x44,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_max_f64 v[4:5], off, s[8:11], s3

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 ; encoding: [0x00,0x00,0x44,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_max_f64 v[4:5], off, s[8:11], s3

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x44,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 offset:7

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x46,0xe1,0x00,0x04,0x02,0x03]
buffer_atomic_max_f64 v[4:5], off, s[8:11], s3 offset:4095 slc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_f64 v1, v[2:3] offset:65535 ; encoding: [0xff,0xff,0xb8,0xd8,0x01,0x02,0x00,0x00]
ds_add_f64 v1, v[2:3] offset:65535

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_f64 v255, v[2:3] offset:65535 ; encoding: [0xff,0xff,0xb8,0xd8,0xff,0x02,0x00,0x00]
ds_add_f64 v255, v[2:3] offset:65535

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_f64 v1, v[254:255] offset:65535 ; encoding: [0xff,0xff,0xb8,0xd8,0x01,0xfe,0x00,0x00]
ds_add_f64 v1, v[254:255] offset:65535

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_f64 v1, v[2:3] ; encoding: [0x00,0x00,0xb8,0xd8,0x01,0x02,0x00,0x00]
ds_add_f64 v1, v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_f64 v1, v[2:3] ; encoding: [0x00,0x00,0xb8,0xd8,0x01,0x02,0x00,0x00]
ds_add_f64 v1, v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_f64 v1, v[2:3] offset:4 ; encoding: [0x04,0x00,0xb8,0xd8,0x01,0x02,0x00,0x00]
ds_add_f64 v1, v[2:3] offset:4

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_f64 v1, v[2:3] offset:65535 gds ; encoding: [0xff,0xff,0xb9,0xd8,0x01,0x02,0x00,0x00]
ds_add_f64 v1, v[2:3] offset:65535 gds

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:65535 ; encoding: [0xff,0xff,0xf8,0xd8,0x01,0x02,0x00,0x04]
ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:65535

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_rtn_f64 v[254:255], v1, v[2:3] offset:65535 ; encoding: [0xff,0xff,0xf8,0xd8,0x01,0x02,0x00,0xfe]
ds_add_rtn_f64 v[254:255], v1, v[2:3] offset:65535

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_rtn_f64 v[4:5], v255, v[2:3] offset:65535 ; encoding: [0xff,0xff,0xf8,0xd8,0xff,0x02,0x00,0x04]
ds_add_rtn_f64 v[4:5], v255, v[2:3] offset:65535

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_rtn_f64 v[4:5], v1, v[254:255] offset:65535 ; encoding: [0xff,0xff,0xf8,0xd8,0x01,0xfe,0x00,0x04]
ds_add_rtn_f64 v[4:5], v1, v[254:255] offset:65535

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_rtn_f64 v[4:5], v1, v[2:3] ; encoding: [0x00,0x00,0xf8,0xd8,0x01,0x02,0x00,0x04]
ds_add_rtn_f64 v[4:5], v1, v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_rtn_f64 v[4:5], v1, v[2:3] ; encoding: [0x00,0x00,0xf8,0xd8,0x01,0x02,0x00,0x04]
ds_add_rtn_f64 v[4:5], v1, v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:4 ; encoding: [0x04,0x00,0xf8,0xd8,0x01,0x02,0x00,0x04]
ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:4

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:65535 gds ; encoding: [0xff,0xff,0xf9,0xd8,0x01,0x02,0x00,0x04]
ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:65535 gds

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_add_f64 v[0:1], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x3c,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_add_f64 v[0:1], v[2:3] offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_add_f64 v[254:255], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x3c,0xdd,0xfe,0x02,0x00,0x00]
flat_atomic_add_f64 v[254:255], v[2:3] offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_add_f64 v[0:1], v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x3c,0xdd,0x00,0xfe,0x00,0x00]
flat_atomic_add_f64 v[0:1], v[254:255] offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_add_f64 v[0:1], v[2:3] ; encoding: [0x00,0x00,0x3c,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_add_f64 v[0:1], v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_add_f64 v[0:1], v[2:3] ; encoding: [0x00,0x00,0x3c,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_add_f64 v[0:1], v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_add_f64 v[0:1], v[2:3] offset:7 ; encoding: [0x07,0x00,0x3c,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_add_f64 v[0:1], v[2:3] offset:7

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_add_f64 v[0:1], v[0:1], v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x3d,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_add_f64 v[0:1], v[0:1], v[2:3] offset:4095 glc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_add_f64 v[0:1], v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x3e,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_add_f64 v[0:1], v[2:3] offset:4095 slc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_min_f64 v[0:1], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x40,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_min_f64 v[0:1], v[2:3] offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_min_f64 v[254:255], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x40,0xdd,0xfe,0x02,0x00,0x00]
flat_atomic_min_f64 v[254:255], v[2:3] offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_min_f64 v[0:1], v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x40,0xdd,0x00,0xfe,0x00,0x00]
flat_atomic_min_f64 v[0:1], v[254:255] offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_min_f64 v[0:1], v[2:3] ; encoding: [0x00,0x00,0x40,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_min_f64 v[0:1], v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_min_f64 v[0:1], v[2:3] ; encoding: [0x00,0x00,0x40,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_min_f64 v[0:1], v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_min_f64 v[0:1], v[2:3] offset:7 ; encoding: [0x07,0x00,0x40,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_min_f64 v[0:1], v[2:3] offset:7

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_min_f64 v[0:1], v[0:1], v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x41,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_min_f64 v[0:1], v[0:1], v[2:3] offset:4095 glc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_min_f64 v[0:1], v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x42,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_min_f64 v[0:1], v[2:3] offset:4095 slc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_max_f64 v[0:1], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x44,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_max_f64 v[0:1], v[2:3] offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_max_f64 v[254:255], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x44,0xdd,0xfe,0x02,0x00,0x00]
flat_atomic_max_f64 v[254:255], v[2:3] offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_max_f64 v[0:1], v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x44,0xdd,0x00,0xfe,0x00,0x00]
flat_atomic_max_f64 v[0:1], v[254:255] offset:4095

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_max_f64 v[0:1], v[2:3] ; encoding: [0x00,0x00,0x44,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_max_f64 v[0:1], v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_max_f64 v[0:1], v[2:3] ; encoding: [0x00,0x00,0x44,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_max_f64 v[0:1], v[2:3]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_max_f64 v[0:1], v[2:3] offset:7 ; encoding: [0x07,0x00,0x44,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_max_f64 v[0:1], v[2:3] offset:7

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_max_f64 v[0:1], v[0:1], v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x45,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_max_f64 v[0:1], v[0:1], v[2:3] offset:4095 glc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: flat_atomic_max_f64 v[0:1], v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x46,0xdd,0x00,0x02,0x00,0x00]
flat_atomic_max_f64 v[0:1], v[2:3] offset:4095 slc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: global_atomic_add_f64 v[0:1], v[2:3], off ; encoding: [0x00,0x80,0x3c,0xdd,0x00,0x02,0x7f,0x00]
global_atomic_add_f64 v[0:1], v[2:3], off

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: global_atomic_min_f64 v[0:1], v[2:3], off ; encoding: [0x00,0x80,0x40,0xdd,0x00,0x02,0x7f,0x00]
global_atomic_min_f64 v[0:1], v[2:3], off

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: global_atomic_max_f64 v[0:1], v[2:3], off ; encoding: [0x00,0x80,0x44,0xdd,0x00,0x02,0x7f,0x00]
global_atomic_max_f64 v[0:1], v[2:3], off

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], v[2:3], v[4:5] ; encoding: [0x02,0x09,0x08,0x08]
v_fmac_f64_e32 v[4:5], v[2:3], v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[254:255], v[2:3], v[4:5] ; encoding: [0x02,0x09,0xfc,0x09]
v_fmac_f64_e32 v[254:255], v[2:3], v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], v[254:255], v[4:5] ; encoding: [0xfe,0x09,0x08,0x08]
v_fmac_f64_e32 v[4:5], v[254:255], v[4:5]

// GFX90A: v_fmac_f64_e32 v[4:5], flat_scratch, v[4:5] ; encoding: [0x66,0x08,0x08,0x08]
// GFX1010: error: instruction not supported on this GPU
// GFX908: error: instruction not supported on this GPU
v_fmac_f64_e32 v[4:5], flat_scratch, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], vcc, v[4:5] ; encoding: [0x6a,0x08,0x08,0x08]
v_fmac_f64_e32 v[4:5], vcc, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], exec, v[4:5] ; encoding: [0x7e,0x08,0x08,0x08]
v_fmac_f64_e32 v[4:5], exec, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], 0, v[4:5] ; encoding: [0x80,0x08,0x08,0x08]
v_fmac_f64_e32 v[4:5], 0, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], -1, v[4:5] ; encoding: [0xc1,0x08,0x08,0x08]
v_fmac_f64_e32 v[4:5], -1, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], 0.5, v[4:5] ; encoding: [0xf0,0x08,0x08,0x08]
v_fmac_f64_e32 v[4:5], 0.5, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], -4.0, v[4:5] ; encoding: [0xf7,0x08,0x08,0x08]
v_fmac_f64_e32 v[4:5], -4.0, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], 0xaf123456, v[4:5] ; encoding: [0xff,0x08,0x08,0x08,0x56,0x34,0x12,0xaf]
v_fmac_f64_e32 v[4:5], 0xaf123456, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], 0x3f717273, v[4:5] ; encoding: [0xff,0x08,0x08,0x08,0x73,0x72,0x71,0x3f]
v_fmac_f64_e32 v[4:5], 0x3f717273, v[4:5]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e32 v[4:5], v[2:3], v[254:255] ; encoding: [0x02,0xfd,0x09,0x08]
v_fmac_f64_e32 v[4:5], v[2:3], v[254:255]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0x02,0x11,0x02,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[254:255], v[2:3], v[8:9] ; encoding: [0xfe,0x00,0x04,0xd1,0x02,0x11,0x02,0x00]
v_fmac_f64_e64 v[254:255], v[2:3], v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[254:255], v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0xfe,0x11,0x02,0x00]
v_fmac_f64_e64 v[4:5], v[254:255], v[8:9]

// GFX90A: v_fmac_f64_e64 v[4:5], flat_scratch, v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0x66,0x10,0x02,0x00]
// GFX1010: error: instruction not supported on this GPU
// GFX908: error: instruction not supported on this GPU
v_fmac_f64_e64 v[4:5], flat_scratch, v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], vcc, v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0x6a,0x10,0x02,0x00]
v_fmac_f64_e64 v[4:5], vcc, v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], exec, v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0x7e,0x10,0x02,0x00]
v_fmac_f64_e64 v[4:5], exec, v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], 0, v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0x80,0x10,0x02,0x00]
v_fmac_f64_e64 v[4:5], 0, v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], -1, v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0xc1,0x10,0x02,0x00]
v_fmac_f64_e64 v[4:5], -1, v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], 0.5, v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0xf0,0x10,0x02,0x00]
v_fmac_f64_e64 v[4:5], 0.5, v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], -4.0, v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0xf7,0x10,0x02,0x00]
v_fmac_f64_e64 v[4:5], -4.0, v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], v[254:255] ; encoding: [0x04,0x00,0x04,0xd1,0x02,0xfd,0x03,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], v[254:255]

// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], flat_scratch ; encoding: [0x04,0x00,0x04,0xd1,0x02,0xcd,0x00,0x00]
// GFX1010: error: instruction not supported on this GPU
// GFX908: error: instruction not supported on this GPU
v_fmac_f64_e64 v[4:5], v[2:3], flat_scratch

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], vcc ; encoding: [0x04,0x00,0x04,0xd1,0x02,0xd5,0x00,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], vcc

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], exec ; encoding: [0x04,0x00,0x04,0xd1,0x02,0xfd,0x00,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], exec

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], 0 ; encoding: [0x04,0x00,0x04,0xd1,0x02,0x01,0x01,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], 0

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], -1 ; encoding: [0x04,0x00,0x04,0xd1,0x02,0x83,0x01,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], -1

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], 0.5 ; encoding: [0x04,0x00,0x04,0xd1,0x02,0xe1,0x01,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], 0.5

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], -4.0 ; encoding: [0x04,0x00,0x04,0xd1,0x02,0xef,0x01,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], -4.0

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], -v[2:3], v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0x02,0x11,0x02,0x20]
v_fmac_f64_e64 v[4:5], -v[2:3], v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], -v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0x02,0x11,0x02,0x40]
v_fmac_f64_e64 v[4:5], v[2:3], -v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], -v[2:3], -v[8:9] ; encoding: [0x04,0x00,0x04,0xd1,0x02,0x11,0x02,0x60]
v_fmac_f64_e64 v[4:5], -v[2:3], -v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], |v[2:3]|, v[8:9] ; encoding: [0x04,0x01,0x04,0xd1,0x02,0x11,0x02,0x00]
v_fmac_f64_e64 v[4:5], |v[2:3]|, v[8:9]

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], |v[8:9]| ; encoding: [0x04,0x02,0x04,0xd1,0x02,0x11,0x02,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], |v[8:9]|

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], |v[2:3]|, |v[8:9]| ; encoding: [0x04,0x03,0x04,0xd1,0x02,0x11,0x02,0x00]
v_fmac_f64_e64 v[4:5], |v[2:3]|, |v[8:9]|

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] clamp ; encoding: [0x04,0x80,0x04,0xd1,0x02,0x11,0x02,0x00]
v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] clamp

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] mul:2 ; encoding: [0x04,0x00,0x04,0xd1,0x02,0x11,0x02,0x08]
v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] mul:2

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] mul:4 ; encoding: [0x04,0x00,0x04,0xd1,0x02,0x11,0x02,0x10]
v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] mul:4

// NOT-GFX90A: error: instruction not supported on this GPU
// GFX90A: v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] div:2 ; encoding: [0x04,0x00,0x04,0xd1,0x02,0x11,0x02,0x18]
v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] div:2

// GFX90A: v_mul_legacy_f32 v5, v1, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x05,0x02,0x00]
v_mul_legacy_f32_e64 v5, v1, v2

// GFX90A: v_mul_legacy_f32 v255, v1, v2 ; encoding: [0xff,0x00,0xa1,0xd2,0x01,0x05,0x02,0x00]
v_mul_legacy_f32_e64 v255, v1, v2

// GFX90A: v_mul_legacy_f32 v5, v255, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0xff,0x05,0x02,0x00]
v_mul_legacy_f32_e64 v5, v255, v2

// GFX90A: v_mul_legacy_f32 v5, s1, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, s1, v2

// GFX90A: v_mul_legacy_f32 v5, s101, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x65,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, s101, v2

// GFX90A: v_mul_legacy_f32 v5, vcc_lo, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x6a,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, vcc_lo, v2

// GFX90A: v_mul_legacy_f32 v5, vcc_hi, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x6b,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, vcc_hi, v2

// GFX90A: v_mul_legacy_f32 v5, m0, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x7c,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, m0, v2

// GFX90A: v_mul_legacy_f32 v5, exec_lo, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x7e,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, exec_lo, v2

// GFX90A: v_mul_legacy_f32 v5, exec_hi, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x7f,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, exec_hi, v2

// GFX90A: v_mul_legacy_f32 v5, 0, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x80,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, 0, v2

// GFX90A: v_mul_legacy_f32 v5, -1, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0xc1,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, -1, v2

// GFX90A: v_mul_legacy_f32 v5, 0.5, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0xf0,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, 0.5, v2

// GFX90A: v_mul_legacy_f32 v5, -4.0, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0xf7,0x04,0x02,0x00]
v_mul_legacy_f32_e64 v5, -4.0, v2

// GFX90A: v_mul_legacy_f32 v5, v1, v255 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0xff,0x03,0x00]
v_mul_legacy_f32_e64 v5, v1, v255

// GFX90A: v_mul_legacy_f32 v5, v1, s2 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x05,0x00,0x00]
v_mul_legacy_f32_e64 v5, v1, s2

// GFX90A: v_mul_legacy_f32 v5, v1, s101 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0xcb,0x00,0x00]
v_mul_legacy_f32_e64 v5, v1, s101

// GFX90A: v_mul_legacy_f32 v5, v1, vcc_lo ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0xd5,0x00,0x00]
v_mul_legacy_f32_e64 v5, v1, vcc_lo

// GFX90A: v_mul_legacy_f32 v5, v1, vcc_hi ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0xd7,0x00,0x00]
v_mul_legacy_f32_e64 v5, v1, vcc_hi

// GFX90A: v_mul_legacy_f32 v5, v1, m0 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0xf9,0x00,0x00]
v_mul_legacy_f32_e64 v5, v1, m0

// GFX90A: v_mul_legacy_f32 v5, v1, exec_lo ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0xfd,0x00,0x00]
v_mul_legacy_f32_e64 v5, v1, exec_lo

// GFX90A: v_mul_legacy_f32 v5, v1, exec_hi ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0xff,0x00,0x00]
v_mul_legacy_f32_e64 v5, v1, exec_hi

// GFX90A: v_mul_legacy_f32 v5, v1, 0 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x01,0x01,0x00]
v_mul_legacy_f32_e64 v5, v1, 0

// GFX90A: v_mul_legacy_f32 v5, v1, -1 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x83,0x01,0x00]
v_mul_legacy_f32_e64 v5, v1, -1

// GFX90A: v_mul_legacy_f32 v5, v1, 0.5 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0xe1,0x01,0x00]
v_mul_legacy_f32_e64 v5, v1, 0.5

// GFX90A: v_mul_legacy_f32 v5, v1, -4.0 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0xef,0x01,0x00]
v_mul_legacy_f32_e64 v5, v1, -4.0

// GFX90A: v_mul_legacy_f32 v5, -v1, v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x05,0x02,0x20]
v_mul_legacy_f32_e64 v5, -v1, v2

// GFX90A: v_mul_legacy_f32 v5, v1, -v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x05,0x02,0x40]
v_mul_legacy_f32_e64 v5, v1, -v2

// GFX90A: v_mul_legacy_f32 v5, -v1, -v2 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x05,0x02,0x60]
v_mul_legacy_f32_e64 v5, -v1, -v2

// GFX90A: v_mul_legacy_f32 v5, |v1|, v2 ; encoding: [0x05,0x01,0xa1,0xd2,0x01,0x05,0x02,0x00]
v_mul_legacy_f32_e64 v5, |v1|, v2

// GFX90A: v_mul_legacy_f32 v5, v1, |v2| ; encoding: [0x05,0x02,0xa1,0xd2,0x01,0x05,0x02,0x00]
v_mul_legacy_f32_e64 v5, v1, |v2|

// GFX90A: v_mul_legacy_f32 v5, |v1|, |v2| ; encoding: [0x05,0x03,0xa1,0xd2,0x01,0x05,0x02,0x00]
v_mul_legacy_f32_e64 v5, |v1|, |v2|

// GFX90A: v_mul_legacy_f32 v5, v1, v2 clamp ; encoding: [0x05,0x80,0xa1,0xd2,0x01,0x05,0x02,0x00]
v_mul_legacy_f32_e64 v5, v1, v2 clamp

// GFX90A: v_mul_legacy_f32 v5, v1, v2 mul:2 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x05,0x02,0x08]
v_mul_legacy_f32_e64 v5, v1, v2 mul:2

// GFX90A: v_mul_legacy_f32 v5, v1, v2 mul:4 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x05,0x02,0x10]
v_mul_legacy_f32_e64 v5, v1, v2 mul:4

// GFX90A: v_mul_legacy_f32 v5, v1, v2 div:2 ; encoding: [0x05,0x00,0xa1,0xd2,0x01,0x05,0x02,0x18]
v_mul_legacy_f32_e64 v5, v1, v2 div:2

// GFX90A: v_xor_b32_dpp v6, v29, v27  row_newbcast:0 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x36,0x0c,0x2a,0x1d,0x50,0x01,0xff]
// NOT-GFX90A: error: not a valid operand.
v_xor_b32 v6, v29, v27 row_newbcast:0

// GFX90A: v_xor_b32_dpp v6, v29, v27  row_newbcast:7 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x36,0x0c,0x2a,0x1d,0x57,0x01,0xff]
// NOT-GFX90A: error: not a valid operand.
v_xor_b32 v6, v29, v27 row_newbcast:7

// GFX90A: v_xor_b32_dpp v6, v29, v27  row_newbcast:15 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x36,0x0c,0x2a,0x1d,0x5f,0x01,0xff]
// NOT-GFX90A: error: not a valid operand.
v_xor_b32 v6, v29, v27 row_newbcast:15

// GFX90A: buffer_atomic_add_f32 v0, v2, s[4:7], 0 idxen glc ; encoding: [0x00,0x60,0x34,0xe1,0x02,0x00,0x01,0x80]
// GFX1010: error: instruction not supported on this GPU
// GFX908: error: instruction must not use glc
buffer_atomic_add_f32 v0, v2, s[4:7], 0 idxen glc

// GFX90A: buffer_atomic_add_f32 v0, v2, s[4:7], 0 idxen glc ; encoding: [0x00,0x60,0x34,0xe1,0x02,0x00,0x01,0x80]
// GFX1010: error: instruction not supported on this GPU
// GFX908: error: instruction must not use glc
buffer_atomic_add_f32 v0, v2, s[4:7], 0 idxen glc

// GFX90A: buffer_atomic_pk_add_f16 v0, v2, s[4:7], 0 idxen glc ; encoding: [0x00,0x60,0x38,0xe1,0x02,0x00,0x01,0x80]
// GFX1010: error: instruction not supported on this GPU
// GFX908: error: instruction must not use glc
buffer_atomic_pk_add_f16 v0, v2, s[4:7], 0 idxen glc

// GFX90A: global_atomic_add_f32 v0, v[0:1], v2, off glc ; encoding: [0x00,0x80,0x35,0xdd,0x00,0x02,0x7f,0x00]
// GFX1010: error: instruction not supported on this GPU
// GFX908: error: operands are not valid for this GPU or mode
global_atomic_add_f32 v0, v[0:1], v2, off glc

// GFX90A: global_atomic_pk_add_f16 v0, v[0:1], v2, off glc ; encoding: [0x00,0x80,0x39,0xdd,0x00,0x02,0x7f,0x00]
// GFX1010: error: instruction not supported on this GPU
// GFX908: error: operands are not valid for this GPU or mode
global_atomic_pk_add_f16 v0, v[0:1], v2, off glc

// GFX90A: global_atomic_add_f64 v[0:1], v[0:1], v[2:3], off glc ; encoding: [0x00,0x80,0x3d,0xdd,0x00,0x02,0x7f,0x00]
// NOT-GFX90A: error: instruction not supported on this GPU
global_atomic_add_f64 v[0:1], v[0:1], v[2:3], off glc

// GFX90A: global_atomic_max_f64 v[0:1], v[0:1], v[2:3], off glc ; encoding: [0x00,0x80,0x45,0xdd,0x00,0x02,0x7f,0x00]
// NOT-GFX90A: error: instruction not supported on this GPU
global_atomic_max_f64 v[0:1], v[0:1], v[2:3], off glc

// GFX90A: global_atomic_min_f64 v[0:1], v[0:1], v[2:3], off glc ; encoding: [0x00,0x80,0x41,0xdd,0x00,0x02,0x7f,0x00]
// NOT-GFX90A: error: instruction not supported on this GPU
global_atomic_min_f64 v[0:1], v[0:1], v[2:3], off glc

// GFX90A: flat_atomic_add_f64 v[0:1], v[0:1], v[2:3] glc ; encoding: [0x00,0x00,0x3d,0xdd,0x00,0x02,0x00,0x00]
// NOT-GFX90A: error: instruction not supported on this GPU
flat_atomic_add_f64 v[0:1], v[0:1], v[2:3] glc

// GFX90A: flat_atomic_max_f64 v[0:1], v[0:1], v[2:3] glc ; encoding: [0x00,0x00,0x45,0xdd,0x00,0x02,0x00,0x00]
// NOT-GFX90A: error: instruction not supported on this GPU
flat_atomic_max_f64 v[0:1], v[0:1], v[2:3] glc

// GFX90A: flat_atomic_min_f64 v[0:1], v[0:1], v[2:3] glc ; encoding: [0x00,0x00,0x41,0xdd,0x00,0x02,0x00,0x00]
// NOT-GFX90A: error: instruction not supported on this GPU
flat_atomic_min_f64 v[0:1], v[0:1], v[2:3] glc

// GFX90A: global_atomic_add_f32  v0, v[0:1], v2, off glc ; encoding: [0x00,0x80,0x35,0xdd,0x00,0x02,0x7f,0x00]
// GFX908: error: operands are not valid for this GPU or mode
// GFX1010: error: instruction not supported on this GPU
global_atomic_add_f32 v0, v[0:1], v2, off glc

// GFX90A: global_atomic_add_f32  v[0:1], v2, off  ; encoding: [0x00,0x80,0x34,0xdd,0x00,0x02,0x7f,0x00]
// GFX1010: error: instruction not supported on this GPU
global_atomic_add_f32 v[0:1], v2, off

// GFX90A: global_atomic_add_f32  v0, v2, s[0:1]   ; encoding: [0x00,0x80,0x34,0xdd,0x00,0x02,0x00,0x00]
// GFX1010: error: instruction not supported on this GPU
global_atomic_add_f32 v0, v2, s[0:1]

// GFX90A: global_atomic_add_f32  v1, v0, v2, s[0:1] glc
// GFX908: error: operands are not valid for this GPU or mode
// GFX1010: error: instruction not supported on this GPU
global_atomic_add_f32 v1, v0, v2, s[0:1] glc ; encoding: [0x00,0x80,0x35,0xdd,0x00,0x02,0x00,0x01]

// GFX908: error: operands are not valid for this GPU or mode
// GFX1010: error: instruction not supported on this GPU
// GFX90A: global_atomic_pk_add_f16  v0, v[0:1], v2, off glc ; encoding: [0x00,0x80,0x39,0xdd,0x00,0x02,0x7f,0x00]
global_atomic_pk_add_f16 v0, v[0:1], v2, off glc
