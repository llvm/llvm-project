// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s --check-prefix=SICI
// RUN: llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck %s --check-prefix=SICI
// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefix=VI

//===----------------------------------------------------------------------===//
// Generic Checks
//===----------------------------------------------------------------------===//

// src0 sgpr
v_cmp_lt_f32 vcc, s2, v4
// SICI: v_cmp_lt_f32_e32 vcc, s2, v4 ; encoding: [0x02,0x08,0x02,0x7c]
// VI:   v_cmp_lt_f32_e32 vcc, s2, v4 ; encoding: [0x02,0x08,0x82,0x7c]

// src0 inline immediate
v_cmp_lt_f32 vcc, 0, v4
// SICI: v_cmp_lt_f32_e32 vcc, 0, v4 ; encoding: [0x80,0x08,0x02,0x7c]
// VI:   v_cmp_lt_f32_e32 vcc, 0, v4 ; encoding: [0x80,0x08,0x82,0x7c]

// src0 literal
v_cmp_lt_f32 vcc, 10.0, v4
// SICI: v_cmp_lt_f32_e32 vcc, 0x41200000, v4 ; encoding: [0xff,0x08,0x02,0x7c,0x00,0x00,0x20,0x41]
// VI:   v_cmp_lt_f32_e32 vcc, 0x41200000, v4 ; encoding: [0xff,0x08,0x82,0x7c,0x00,0x00,0x20,0x41]

// src0, src1 max vgpr
v_cmp_lt_f32 vcc, v255, v255
// SICI: v_cmp_lt_f32_e32 vcc, v255, v255 ; encoding: [0xff,0xff,0x03,0x7c]
// VI:   v_cmp_lt_f32_e32 vcc, v255, v255 ; encoding: [0xff,0xff,0x83,0x7c]

// force 32-bit encoding
v_cmp_lt_f32_e32 vcc, v2, v4
// SICI: v_cmp_lt_f32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x02,0x7c]
// VI:   v_cmp_lt_f32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x82,0x7c]


//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

v_cmp_f_f32 vcc, v2, v4
// SICI: v_cmp_f_f32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x00,0x7c]
// VI:   v_cmp_f_f32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x80,0x7c]

v_cmp_lt_f32 vcc, v2, v4
// SICI: v_cmp_lt_f32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x02,0x7c]
// VI:   v_cmp_lt_f32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x82,0x7c]

// TODO: Add tests for the rest of v_cmp_*_f32
// TODO: Add tests for v_cmpx_*_f32

v_cmp_f_f64 vcc, v[2:3], v[4:5]
// SICI: v_cmp_f_f64_e32 vcc, v[2:3], v[4:5] ; encoding: [0x02,0x09,0x40,0x7c]
// VI:   v_cmp_f_f64_e32 vcc, v[2:3], v[4:5] ; encoding: [0x02,0x09,0xc0,0x7c]

// TODO: Add tests for the rest of v_cmp_*_f64
// TODO: Add tests for the rest of the floating-point comparision instructions.

v_cmp_f_i32 vcc, v2, v4
// SICI: v_cmp_f_i32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x00,0x7d]
// VI:   v_cmp_f_i32_e32 vcc, v2, v4 ; encoding: [0x02,0x09,0x80,0x7d]

// TODO: Add test for the rest of v_cmp_*_i32

v_cmp_f_i64 vcc, v[2:3], v[4:5]
// SICI: v_cmp_f_i64_e32 vcc, v[2:3], v[4:5] ; encoding: [0x02,0x09,0x40,0x7d]
// VI:   v_cmp_f_i64_e32 vcc, v[2:3], v[4:5] ; encoding: [0x02,0x09,0xc0,0x7d]

// TODO: Add tests for the rest of the instructions.
