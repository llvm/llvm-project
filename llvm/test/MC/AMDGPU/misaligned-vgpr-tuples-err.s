// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefixes=GFX90A --implicit-check-not=error: %s

v_add_f64 v[1:2], v[1:2], v[1:2]
// GFX90A: error: invalid register class: vgpr tuples must be 64 bit aligned

global_load_dwordx3 v[1:3], v[0:1], off
// GFX90A: error: invalid register class: vgpr tuples must be 64 bit aligned

global_load_dwordx4 v[1:4], v[0:1], off
// GFX90A: error: invalid register class: vgpr tuples must be 64 bit aligned

image_load v[1:5], v2, s[0:7] dmask:0xf unorm
// GFX90A: error: invalid register class: vgpr tuples must be 64 bit aligned

v_mfma_f32_32x32x8f16 a[0:15], a[1:2], v[0:1], a[0:15]
// GFX90A: error: invalid register class: vgpr tuples must be 64 bit aligned

v_mfma_i32_4x4x4i8 a[1:4], a0, v1, 2
// GFX90A: error: invalid register class: vgpr tuples must be 64 bit aligned

v_mfma_f32_16x16x1f32 a[0:15], a0, v1, a[17:32]
// GFX90A: error: invalid register class: vgpr tuples must be 64 bit aligned

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[33:64]
// GFX90A: error: invalid register class: vgpr tuples must be 64 bit aligned
