// RUN: llvm-mc -triple=amdgpu9.50 -show-encoding %s | FileCheck %s

// CHECK: v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23] cbsz:4 blgp:4 ; encoding: [0x00,0x04,0xad,0xd3,0x04,0x19,0x52,0x84]
v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23] cbsz:4 blgp:4

// CHECK: v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:17], a[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:2 blgp:4 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x8a,0xad,0xd3,0x04,0x19,0x52,0x9c]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:17], a[20:23], v24, v25 cbsz:2 blgp:4

// CHECK: v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 op_sel_hi:[0,0,0] cbsz:4 blgp:2 ; encoding: [0x00,0x00,0xac,0xd3,0x18,0x33,0x02,0x00,0x00,0x0c,0xad,0xd3,0x04,0x19,0x52,0x44]
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[20:23], v24, v25 cbsz:4 blgp:2
