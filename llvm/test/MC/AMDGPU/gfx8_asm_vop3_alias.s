// RUN: llvm-mc -triple=amdgpu8.02 -show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple=amdgpu9.00 -show-encoding %s | FileCheck %s
// RUN: not llvm-mc -triple=amdgpu11.00 %s 2>&1 | FileCheck --check-prefix=GFX11-ERR --implicit-check-not=error: %s

// v_mul_lo_i32 was removed in GFX8 because it is identical to v_mul_lo_u32.
// It is accepted as an alias on GFX8 and GFX9 only.

v_mul_lo_i32 v5, v1, v2
// CHECK: v_mul_lo_u32 v5, v1, v2                 ; encoding: [0x05,0x00,0x85,0xd2,0x01,0x05,0x02,0x00]
// GFX11-ERR: :[[@LINE-2]]:1: error: instruction not supported on this GPU (gfx1100): v_mul_lo_i32

v_mul_lo_i32 v5, s1, 0.5
// CHECK: v_mul_lo_u32 v5, s1, 0.5                ; encoding: [0x05,0x00,0x85,0xd2,0x01,0xe0,0x01,0x00]
// GFX11-ERR: :[[@LINE-2]]:1: error: instruction not supported on this GPU (gfx1100): v_mul_lo_i32

v_mul_lo_u32 v5, v1, v2
// CHECK: v_mul_lo_u32 v5, v1, v2                 ; encoding: [0x05,0x00,0x85,0xd2,0x01,0x05,0x02,0x00]
