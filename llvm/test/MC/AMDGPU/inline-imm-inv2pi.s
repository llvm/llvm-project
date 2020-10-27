// RUN: llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck -check-prefix=SI %s
// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=VI %s

// The value inv2pi should not assert on any targets, but is
// printed differently depending on whether it's a legal inline
// immediate or not.

// SI: v_cvt_f32_f16_e32 v0, 0x3118 ; encoding: [0xff,0x16,0x00,0x7e,0x18,0x31,0x00,0x00]
// VI: v_cvt_f32_f16_e32 v0, 0.15915494 ; encoding: [0xf8,0x16,0x00,0x7e]
v_cvt_f32_f16_e32 v0, 0x3118
