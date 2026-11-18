// RUN: llvm-mc -triple=amdgpu12.50 -show-encoding %s | FileCheck %s

// CHECK: v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:35], v[40:47] matrix_a_fmt:MATRIX_FMT_FP6 matrix_b_fmt:MATRIX_FMT_FP4 ; encoding: [0x00,0x50,0x33,0xcc,0x08,0x31,0xa2,0x04]
v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:35], v[40:47] matrix_a_fmt:MATRIX_FMT_FP6 matrix_b_fmt:MATRIX_FMT_FP4

// CHECK: v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:19], v[24:39], v[40:47], v1, v2 matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP6 ; encoding: [0x00,0x00,0x35,0xcc,0x01,0x05,0x02,0x02,0x00,0x20,0x33,0xcc,0x08,0x31,0xa2,0x14]
v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:19], v[24:39], v[40:47], v1, v2 matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP6

// CHECK: v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4 ; encoding: [0x00,0x60,0x33,0xcc,0x08,0x31,0xa2,0x04]
v_wmma_f32_16x16x128_f8f6f4 v[0:7], v[8:23], v[24:39], v[40:47] matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
