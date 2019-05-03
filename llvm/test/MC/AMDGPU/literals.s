// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SI --check-prefix=SICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SI --check-prefix=SICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI --check-prefix=CIVI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=CIVI --check-prefix=GFX89
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=CIVI --check-prefix=GFX89 --check-prefix=GFX9

// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOGCN --check-prefix=NOSI --check-prefix=NOSICI --check-prefix=NOSICIVI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOGCN --check-prefix=NOSI --check-prefix=NOSICI --check-prefix=NOSICIVI
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOGCN --check-prefix=NOSICI --check-prefix=NOCIVI --check-prefix=NOSICIVI
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOGCN --check-prefix=NOSICIVI --check-prefix=NOVI --check-prefix=NOGFX89
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOGCN --check-prefix=NOGFX89 --check-prefix=NOGFX9

//---------------------------------------------------------------------------//
// fp literal, expected fp operand
//---------------------------------------------------------------------------//

// SICI: v_fract_f64_e32 v[0:1], 0.5 ; encoding: [0xf0,0x7c,0x00,0x7e]
// GFX89: v_fract_f64_e32 v[0:1], 0.5 ; encoding: [0xf0,0x64,0x00,0x7e]
v_fract_f64 v[0:1], 0.5

// SICI: v_sqrt_f64_e32 v[0:1], -4.0 ; encoding: [0xf7,0x68,0x00,0x7e]
// GFX89: v_sqrt_f64_e32 v[0:1], -4.0 ; encoding: [0xf7,0x50,0x00,0x7e]
v_sqrt_f64 v[0:1], -4.0

// SICI: v_log_clamp_f32_e32 v1, 0.5 ; encoding: [0xf0,0x4c,0x02,0x7e]
// NOGFX89: error: instruction not supported on this GPU
v_log_clamp_f32 v1, 0.5

// SICI: v_fract_f64_e32 v[0:1], 0.5 ; encoding: [0xf0,0x7c,0x00,0x7e]
// GFX89: v_fract_f64_e32 v[0:1], 0.5 ; encoding: [0xf0,0x64,0x00,0x7e]
v_fract_f64 v[0:1], 0.5

// SICI: v_trunc_f32_e32 v0, 0.5 ; encoding: [0xf0,0x42,0x00,0x7e]
// GFX89: v_trunc_f32_e32 v0, 0.5 ; encoding: [0xf0,0x38,0x00,0x7e]
v_trunc_f32 v0, 0.5

// SICI: v_fract_f64_e32 v[0:1], -1.0 ; encoding: [0xf3,0x7c,0x00,0x7e]
// GFX89: v_fract_f64_e32 v[0:1], -1.0 ; encoding: [0xf3,0x64,0x00,0x7e]
v_fract_f64 v[0:1], -1.0

// SICI: v_trunc_f32_e32 v0, -1.0 ; encoding: [0xf3,0x42,0x00,0x7e]
// GFX89: v_trunc_f32_e32 v0, -1.0 ; encoding: [0xf3,0x38,0x00,0x7e]
v_trunc_f32 v0, -1.0

// SICI: v_fract_f64_e32 v[0:1], 4.0 ; encoding: [0xf6,0x7c,0x00,0x7e]
// GFX89: v_fract_f64_e32 v[0:1], 4.0 ; encoding: [0xf6,0x64,0x00,0x7e]
v_fract_f64 v[0:1], 4.0

// SICI: v_trunc_f32_e32 v0, 4.0 ; encoding: [0xf6,0x42,0x00,0x7e]
// GFX89: v_trunc_f32_e32 v0, 4.0 ; encoding: [0xf6,0x38,0x00,0x7e]
v_trunc_f32 v0, 4.0

// SICI: v_fract_f64_e32 v[0:1], 0 ; encoding: [0x80,0x7c,0x00,0x7e]
// GFX89: v_fract_f64_e32 v[0:1], 0 ; encoding: [0x80,0x64,0x00,0x7e]
v_fract_f64 v[0:1], 0.0

// SICI: v_trunc_f32_e32 v0, 0 ; encoding: [0x80,0x42,0x00,0x7e]
// GFX89: v_trunc_f32_e32 v0, 0 ; encoding: [0x80,0x38,0x00,0x7e]
v_trunc_f32 v0, 0.0

// SICI: v_fract_f64_e32 v[0:1], 0x3ff80000 ; encoding: [0xff,0x7c,0x00,0x7e,0x00,0x00,0xf8,0x3f]
// GFX89: v_fract_f64_e32 v[0:1], 0x3ff80000 ; encoding: [0xff,0x64,0x00,0x7e,0x00,0x00,0xf8,0x3f]
v_fract_f64 v[0:1], 1.5

// SICI: v_trunc_f32_e32 v0, 0x3fc00000 ; encoding: [0xff,0x42,0x00,0x7e,0x00,0x00,0xc0,0x3f]
// GFX89: v_trunc_f32_e32 v0, 0x3fc00000 ; encoding: [0xff,0x38,0x00,0x7e,0x00,0x00,0xc0,0x3f]
v_trunc_f32 v0, 1.5

// SICI: v_fract_f64_e32 v[0:1], 0xc00921ca ; encoding: [0xff,0x7c,0x00,0x7e,0xca,0x21,0x09,0xc0]
// GFX89: v_fract_f64_e32 v[0:1], 0xc00921ca ; encoding: [0xff,0x64,0x00,0x7e,0xca,0x21,0x09,0xc0]
v_fract_f64 v[0:1], -3.1415

// SICI: v_trunc_f32_e32 v0, 0xc0490e56 ; encoding: [0xff,0x42,0x00,0x7e,0x56,0x0e,0x49,0xc0]
// GFX89: v_trunc_f32_e32 v0, 0xc0490e56 ; encoding: [0xff,0x38,0x00,0x7e,0x56,0x0e,0x49,0xc0]
v_trunc_f32 v0, -3.1415

// SICI: v_fract_f64_e32 v[0:1], 0x44b52d02 ; encoding: [0xff,0x7c,0x00,0x7e,0x02,0x2d,0xb5,0x44]
// GFX89: v_fract_f64_e32 v[0:1], 0x44b52d02 ; encoding: [0xff,0x64,0x00,0x7e,0x02,0x2d,0xb5,0x44]
v_fract_f64 v[0:1], 100000000000000000000000.0

// SICI: v_trunc_f32_e32 v0, 0x65a96816 ; encoding: [0xff,0x42,0x00,0x7e,0x16,0x68,0xa9,0x65]
// GFX89: v_trunc_f32_e32 v0, 0x65a96816 ; encoding: [0xff,0x38,0x00,0x7e,0x16,0x68,0xa9,0x65]
v_trunc_f32 v0, 100000000000000000000000.0

// SICI: v_fract_f64_e32 v[0:1], 0x416312d0 ; encoding: [0xff,0x7c,0x00,0x7e,0xd0,0x12,0x63,0x41]
// GFX89: v_fract_f64_e32 v[0:1], 0x416312d0 ; encoding: [0xff,0x64,0x00,0x7e,0xd0,0x12,0x63,0x41]
v_fract_f64 v[0:1], 10000000.0

// SICI: v_trunc_f32_e32 v0, 0x4b189680 ; encoding: [0xff,0x42,0x00,0x7e,0x80,0x96,0x18,0x4b]
// GFX89: v_trunc_f32_e32 v0, 0x4b189680 ; encoding: [0xff,0x38,0x00,0x7e,0x80,0x96,0x18,0x4b]
v_trunc_f32 v0, 10000000.0

// SICI: v_fract_f64_e32 v[0:1], 0x47efffff ; encoding: [0xff,0x7c,0x00,0x7e,0xff,0xff,0xef,0x47]
// GFX89: v_fract_f64_e32 v[0:1], 0x47efffff ; encoding: [0xff,0x64,0x00,0x7e,0xff,0xff,0xef,0x47]
v_fract_f64 v[0:1], 3.402823e+38

// SICI: v_trunc_f32_e32 v0, 0x7f7ffffd ; encoding: [0xff,0x42,0x00,0x7e,0xfd,0xff,0x7f,0x7f]
// GFX89: v_trunc_f32_e32 v0, 0x7f7ffffd ; encoding: [0xff,0x38,0x00,0x7e,0xfd,0xff,0x7f,0x7f]
v_trunc_f32 v0, 3.402823e+38

// SICI: v_fract_f64_e32 v[0:1], 0x381fffff ; encoding: [0xff,0x7c,0x00,0x7e,0xff,0xff,0x1f,0x38]
// GFX89: v_fract_f64_e32 v[0:1], 0x381fffff ; encoding: [0xff,0x64,0x00,0x7e,0xff,0xff,0x1f,0x38]
v_fract_f64 v[0:1], 2.3509886e-38

// SICI: v_trunc_f32_e32 v0, 0xffffff ; encoding: [0xff,0x42,0x00,0x7e,0xff,0xff,0xff,0x00]
// GFX89: v_trunc_f32_e32 v0, 0xffffff ; encoding: [0xff,0x38,0x00,0x7e,0xff,0xff,0xff,0x00]
v_trunc_f32 v0, 2.3509886e-38

// SICI: v_fract_f64_e32 v[0:1], 0x3179f623 ; encoding: [0xff,0x7c,0x00,0x7e,0x23,0xf6,0x79,0x31]
// GFX89: v_fract_f64_e32 v[0:1], 0x3179f623 ; encoding: [0xff,0x64,0x00,0x7e,0x23,0xf6,0x79,0x31]
v_fract_f64 v[0:1], 2.3509886e-70

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
v_trunc_f32 v0, 2.3509886e-70

//---------------------------------------------------------------------------//
// fp literal, expected int operand
//---------------------------------------------------------------------------//

// SICI: s_mov_b64 s[0:1], 0.5 ; encoding: [0xf0,0x04,0x80,0xbe]
// GFX89: s_mov_b64 s[0:1], 0.5 ; encoding: [0xf0,0x01,0x80,0xbe]
s_mov_b64_e32 s[0:1], 0.5

// SICI: v_and_b32_e32 v0, 0.5, v1 ; encoding: [0xf0,0x02,0x00,0x36]
// GFX89: v_and_b32_e32 v0, 0.5, v1 ; encoding: [0xf0,0x02,0x00,0x26]
v_and_b32_e32 v0, 0.5, v1

// SICI: v_and_b32_e64 v0, 0.5, v1 ; encoding: [0x00,0x00,0x36,0xd2,0xf0,0x02,0x02,0x00]
// GFX89: v_and_b32_e64 v0, 0.5, v1 ; encoding: [0x00,0x00,0x13,0xd1,0xf0,0x02,0x02,0x00]
v_and_b32_e64 v0, 0.5, v1

// SICI: s_mov_b64 s[0:1], -1.0 ; encoding: [0xf3,0x04,0x80,0xbe]
// GFX89: s_mov_b64 s[0:1], -1.0 ; encoding: [0xf3,0x01,0x80,0xbe]
s_mov_b64_e32 s[0:1], -1.0

// SICI: v_and_b32_e32 v0, -1.0, v1 ; encoding: [0xf3,0x02,0x00,0x36]
// GFX89: v_and_b32_e32 v0, -1.0, v1 ; encoding: [0xf3,0x02,0x00,0x26]
v_and_b32_e32 v0, -1.0, v1

// SICI: v_and_b32_e64 v0, -1.0, v1 ; encoding: [0x00,0x00,0x36,0xd2,0xf3,0x02,0x02,0x00]
// GFX89: v_and_b32_e64 v0, -1.0, v1 ; encoding: [0x00,0x00,0x13,0xd1,0xf3,0x02,0x02,0x00]
v_and_b32_e64 v0, -1.0, v1

// SICI: s_mov_b64 s[0:1], 4.0 ; encoding: [0xf6,0x04,0x80,0xbe]
// GFX89: s_mov_b64 s[0:1], 4.0 ; encoding: [0xf6,0x01,0x80,0xbe]
s_mov_b64_e32 s[0:1], 4.0

// SICI: v_and_b32_e32 v0, 4.0, v1 ; encoding: [0xf6,0x02,0x00,0x36]
// GFX89: v_and_b32_e32 v0, 4.0, v1 ; encoding: [0xf6,0x02,0x00,0x26]
v_and_b32_e32 v0, 4.0, v1

// SICI: v_and_b32_e64 v0, 4.0, v1 ; encoding: [0x00,0x00,0x36,0xd2,0xf6,0x02,0x02,0x00]
// GFX89: v_and_b32_e64 v0, 4.0, v1 ; encoding: [0x00,0x00,0x13,0xd1,0xf6,0x02,0x02,0x00]
v_and_b32_e64 v0, 4.0, v1

// SICI: s_mov_b64 s[0:1], 0 ; encoding: [0x80,0x04,0x80,0xbe]
// GFX89: s_mov_b64 s[0:1], 0 ; encoding: [0x80,0x01,0x80,0xbe]
s_mov_b64_e32 s[0:1], 0.0

// SICI: v_and_b32_e32 v0, 0, v1 ; encoding: [0x80,0x02,0x00,0x36]
// GFX89: v_and_b32_e32 v0, 0, v1 ; encoding: [0x80,0x02,0x00,0x26]
v_and_b32_e32 v0, 0.0, v1

// SICI: v_and_b32_e64 v0, 0, v1 ; encoding: [0x00,0x00,0x36,0xd2,0x80,0x02,0x02,0x00]
// GFX89: v_and_b32_e64 v0, 0, v1 ; encoding: [0x00,0x00,0x13,0xd1,0x80,0x02,0x02,0x00]
v_and_b32_e64 v0, 0.0, v1

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
s_mov_b64_e32 s[0:1], 1.5

// SICI: v_and_b32_e32 v0, 0x3fc00000, v1 ; encoding: [0xff,0x02,0x00,0x36,0x00,0x00,0xc0,0x3f]
// GFX89: v_and_b32_e32 v0, 0x3fc00000, v1 ; encoding: [0xff,0x02,0x00,0x26,0x00,0x00,0xc0,0x3f]
v_and_b32_e32 v0, 1.5, v1

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
s_mov_b64_e32 s[0:1], -3.1415

// SICI: v_and_b32_e32 v0, 0xc0490e56, v1 ; encoding: [0xff,0x02,0x00,0x36,0x56,0x0e,0x49,0xc0]
// GFX89: v_and_b32_e32 v0, 0xc0490e56, v1 ; encoding: [0xff,0x02,0x00,0x26,0x56,0x0e,0x49,0xc0]
v_and_b32_e32 v0, -3.1415, v1

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
s_mov_b64_e32 s[0:1], 100000000000000000000000.0

// SICI: v_and_b32_e32 v0, 0x65a96816, v1 ; encoding: [0xff,0x02,0x00,0x36,0x16,0x68,0xa9,0x65]
// GFX89: v_and_b32_e32 v0, 0x65a96816, v1 ; encoding: [0xff,0x02,0x00,0x26,0x16,0x68,0xa9,0x65]
v_and_b32_e32 v0, 100000000000000000000000.0, v1

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
s_mov_b64_e32 s[0:1], 10000000.0

// SICI: v_and_b32_e32 v0, 0x4b189680, v1 ; encoding: [0xff,0x02,0x00,0x36,0x80,0x96,0x18,0x4b]
// GFX89: v_and_b32_e32 v0, 0x4b189680, v1 ; encoding: [0xff,0x02,0x00,0x26,0x80,0x96,0x18,0x4b]
v_and_b32_e32 v0, 10000000.0, v1

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
s_mov_b64_e32 s[0:1], 3.402823e+38

// SICI: v_and_b32_e32 v0, 0x7f7ffffd, v1 ; encoding: [0xff,0x02,0x00,0x36,0xfd,0xff,0x7f,0x7f]
// GFX89: v_and_b32_e32 v0, 0x7f7ffffd, v1 ; encoding: [0xff,0x02,0x00,0x26,0xfd,0xff,0x7f,0x7f]
v_and_b32_e32 v0, 3.402823e+38, v1

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
s_mov_b64_e32 s[0:1], 2.3509886e-38

// SICI: v_and_b32_e32 v0, 0xffffff, v1 ; encoding: [0xff,0x02,0x00,0x36,0xff,0xff,0xff,0x00]
// GFX89: v_and_b32_e32 v0, 0xffffff, v1 ; encoding: [0xff,0x02,0x00,0x26,0xff,0xff,0xff,0x00]
v_and_b32_e32 v0, 2.3509886e-38, v1

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
s_mov_b64_e32 s[0:1], 2.3509886e-70

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
v_and_b32_e32 v0, 2.3509886e-70, v1

//---------------------------------------------------------------------------//
// int literal, expected fp operand
//---------------------------------------------------------------------------//

// SICI: v_trunc_f32_e32 v0, 0 ; encoding: [0x80,0x42,0x00,0x7e]
// GFX89: v_trunc_f32_e32 v0, 0 ; encoding: [0x80,0x38,0x00,0x7e]
v_trunc_f32_e32 v0, 0

// SICI: v_fract_f64_e32 v[0:1], 0 ; encoding: [0x80,0x7c,0x00,0x7e]
// GFX89: v_fract_f64_e32 v[0:1], 0 ; encoding: [0x80,0x64,0x00,0x7e]
v_fract_f64_e32 v[0:1], 0

// SICI: v_trunc_f32_e64 v0, 0 ; encoding: [0x00,0x00,0x42,0xd3,0x80,0x00,0x00,0x00]
// GFX89: v_trunc_f32_e64 v0, 0 ; encoding: [0x00,0x00,0x5c,0xd1,0x80,0x00,0x00,0x00]
v_trunc_f32_e64 v0, 0

// SICI: v_fract_f64_e64 v[0:1], 0 ; encoding: [0x00,0x00,0x7c,0xd3,0x80,0x00,0x00,0x00]
// GFX89: v_fract_f64_e64 v[0:1], 0 ; encoding: [0x00,0x00,0x72,0xd1,0x80,0x00,0x00,0x00]
v_fract_f64_e64 v[0:1], 0

// SICI: v_trunc_f32_e32 v0, -13 ; encoding: [0xcd,0x42,0x00,0x7e]
// GFX89: v_trunc_f32_e32 v0, -13 ; encoding: [0xcd,0x38,0x00,0x7e]
v_trunc_f32_e32 v0, -13

// SICI: v_fract_f64_e32 v[0:1], -13 ; encoding: [0xcd,0x7c,0x00,0x7e]
// GFX89: v_fract_f64_e32 v[0:1], -13 ; encoding: [0xcd,0x64,0x00,0x7e]
v_fract_f64_e32 v[0:1], -13

// SICI: v_trunc_f32_e64 v0, -13 ; encoding: [0x00,0x00,0x42,0xd3,0xcd,0x00,0x00,0x00]
// GFX89: v_trunc_f32_e64 v0, -13 ; encoding: [0x00,0x00,0x5c,0xd1,0xcd,0x00,0x00,0x00]
v_trunc_f32_e64 v0, -13

// SICI: v_fract_f64_e64 v[0:1], -13 ; encoding: [0x00,0x00,0x7c,0xd3,0xcd,0x00,0x00,0x00]
// GFX89: v_fract_f64_e64 v[0:1], -13 ; encoding: [0x00,0x00,0x72,0xd1,0xcd,0x00,0x00,0x00]
v_fract_f64_e64 v[0:1], -13

// SICI: v_trunc_f32_e32 v0, 35 ; encoding: [0xa3,0x42,0x00,0x7e]
// GFX89: v_trunc_f32_e32 v0, 35 ; encoding: [0xa3,0x38,0x00,0x7e]
v_trunc_f32_e32 v0, 35

// SICI: v_fract_f64_e32 v[0:1], 35 ; encoding: [0xa3,0x7c,0x00,0x7e]
// GFX89: v_fract_f64_e32 v[0:1], 35 ; encoding: [0xa3,0x64,0x00,0x7e]
v_fract_f64_e32 v[0:1], 35

// SICI: v_trunc_f32_e64 v0, 35 ; encoding: [0x00,0x00,0x42,0xd3,0xa3,0x00,0x00,0x00]
// GFX89: v_trunc_f32_e64 v0, 35 ; encoding: [0x00,0x00,0x5c,0xd1,0xa3,0x00,0x00,0x00]
v_trunc_f32_e64 v0, 35

// SICI: v_fract_f64_e64 v[0:1], 35 ; encoding: [0x00,0x00,0x7c,0xd3,0xa3,0x00,0x00,0x00]
// GFX89: v_fract_f64_e64 v[0:1], 35 ; encoding: [0x00,0x00,0x72,0xd1,0xa3,0x00,0x00,0x00]
v_fract_f64_e64 v[0:1], 35

// SICI: v_trunc_f32_e32 v0, 0x4d2 ; encoding: [0xff,0x42,0x00,0x7e,0xd2,0x04,0x00,0x00]
// GFX89: v_trunc_f32_e32 v0, 0x4d2 ; encoding: [0xff,0x38,0x00,0x7e,0xd2,0x04,0x00,0x00]
v_trunc_f32_e32 v0, 1234

// SICI: v_fract_f64_e32 v[0:1], 0x4d2 ; encoding: [0xff,0x7c,0x00,0x7e,0xd2,0x04,0x00,0x00]
// GFX89: v_fract_f64_e32 v[0:1], 0x4d2 ; encoding: [0xff,0x64,0x00,0x7e,0xd2,0x04,0x00,0x00]
v_fract_f64_e32 v[0:1], 1234

// NOSICI: error: invalid literal operand
// NOGFX89: error: invalid literal operand
v_trunc_f32_e64 v0, 1234

// NOSICI: error: invalid literal operand
// NOGFX89: error: invalid literal operand
v_fract_f64_e64 v[0:1], 1234

// SICI: v_trunc_f32_e32 v0, 0xffff2bcf ; encoding: [0xff,0x42,0x00,0x7e,0xcf,0x2b,0xff,0xff]
// GFX89: v_trunc_f32_e32 v0, 0xffff2bcf ; encoding: [0xff,0x38,0x00,0x7e,0xcf,0x2b,0xff,0xff]
v_trunc_f32_e32 v0, -54321

// SICI: v_fract_f64_e32 v[0:1], 0xffff2bcf ; encoding: [0xff,0x7c,0x00,0x7e,0xcf,0x2b,0xff,0xff]
// GFX89: v_fract_f64_e32 v[0:1], 0xffff2bcf ; encoding: [0xff,0x64,0x00,0x7e,0xcf,0x2b,0xff,0xff]
v_fract_f64_e32 v[0:1], -54321

// SICI: v_trunc_f32_e32 v0, 0xdeadbeef ; encoding: [0xff,0x42,0x00,0x7e,0xef,0xbe,0xad,0xde]
// GFX89: v_trunc_f32_e32 v0, 0xdeadbeef ; encoding: [0xff,0x38,0x00,0x7e,0xef,0xbe,0xad,0xde]
v_trunc_f32_e32 v0, 0xdeadbeef

// SICI: v_fract_f64_e32 v[0:1], 0xdeadbeef ; encoding: [0xff,0x7c,0x00,0x7e,0xef,0xbe,0xad,0xde]
// GFX89: v_fract_f64_e32 v[0:1], 0xdeadbeef ; encoding: [0xff,0x64,0x00,0x7e,0xef,0xbe,0xad,0xde]
v_fract_f64_e32 v[0:1], 0xdeadbeef

// SICI: v_trunc_f32_e32 v0, -1 ; encoding: [0xc1,0x42,0x00,0x7e]
// GFX89: v_trunc_f32_e32 v0, -1 ; encoding: [0xc1,0x38,0x00,0x7e]
v_trunc_f32_e32 v0, 0xffffffff

// SICI: v_fract_f64_e32 v[0:1], 0xffffffff ; encoding: [0xff,0x7c,0x00,0x7e,0xff,0xff,0xff,0xff]
// GFX89: v_fract_f64_e32 v[0:1], 0xffffffff ; encoding: [0xff,0x64,0x00,0x7e,0xff,0xff,0xff,0xff]
v_fract_f64_e32 v[0:1], 0xffffffff

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
v_trunc_f32_e32 v0, 0x123456789abcdef0

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
v_fract_f64_e32 v[0:1], 0x123456789abcdef0

// SICI: v_trunc_f32_e32 v0, -1 ; encoding: [0xc1,0x42,0x00,0x7e]
// GFX89: v_trunc_f32_e32 v0, -1 ; encoding: [0xc1,0x38,0x00,0x7e]
v_trunc_f32_e32 v0, 0xffffffffffffffff

// SICI: v_fract_f64_e32 v[0:1], -1 ; encoding: [0xc1,0x7c,0x00,0x7e]
// GFX89: v_fract_f64_e32 v[0:1], -1 ; encoding: [0xc1,0x64,0x00,0x7e]
v_fract_f64_e32 v[0:1], 0xffffffffffffffff

//---------------------------------------------------------------------------//
// int literal, expected int operand
//---------------------------------------------------------------------------//

// SICI: s_mov_b64 s[0:1], 0 ; encoding: [0x80,0x04,0x80,0xbe]
// GFX89: s_mov_b64 s[0:1], 0 ; encoding: [0x80,0x01,0x80,0xbe]
s_mov_b64_e32 s[0:1], 0

// SICI: v_and_b32_e32 v0, 0, v1 ; encoding: [0x80,0x02,0x00,0x36]
// GFX89: v_and_b32_e32 v0, 0, v1 ; encoding: [0x80,0x02,0x00,0x26]
v_and_b32_e32 v0, 0, v1

// SICI: v_and_b32_e64 v0, 0, v1 ; encoding: [0x00,0x00,0x36,0xd2,0x80,0x02,0x02,0x00]
// GFX89: v_and_b32_e64 v0, 0, v1 ; encoding: [0x00,0x00,0x13,0xd1,0x80,0x02,0x02,0x00]
v_and_b32_e64 v0, 0, v1

// SICI: s_mov_b64 s[0:1], -13 ; encoding: [0xcd,0x04,0x80,0xbe]
// GFX89: s_mov_b64 s[0:1], -13 ; encoding: [0xcd,0x01,0x80,0xbe]
s_mov_b64_e32 s[0:1], -13

// SICI: v_and_b32_e32 v0, -13, v1 ; encoding: [0xcd,0x02,0x00,0x36]
// GFX89: v_and_b32_e32 v0, -13, v1 ; encoding: [0xcd,0x02,0x00,0x26]
v_and_b32_e32 v0, -13, v1

// SICI: v_and_b32_e64 v0, -13, v1 ; encoding: [0x00,0x00,0x36,0xd2,0xcd,0x02,0x02,0x00]
// GFX89: v_and_b32_e64 v0, -13, v1 ; encoding: [0x00,0x00,0x13,0xd1,0xcd,0x02,0x02,0x00]
v_and_b32_e64 v0, -13, v1

// SICI: s_mov_b64 s[0:1], 35 ; encoding: [0xa3,0x04,0x80,0xbe]
// GFX89: s_mov_b64 s[0:1], 35 ; encoding: [0xa3,0x01,0x80,0xbe]
s_mov_b64_e32 s[0:1], 35

// SICI: v_and_b32_e32 v0, 35, v1 ; encoding: [0xa3,0x02,0x00,0x36]
// GFX89: v_and_b32_e32 v0, 35, v1 ; encoding: [0xa3,0x02,0x00,0x26]
v_and_b32_e32 v0, 35, v1

// SICI: v_and_b32_e64 v0, 35, v1 ; encoding: [0x00,0x00,0x36,0xd2,0xa3,0x02,0x02,0x00]
// GFX89: v_and_b32_e64 v0, 35, v1 ; encoding: [0x00,0x00,0x13,0xd1,0xa3,0x02,0x02,0x00]
v_and_b32_e64 v0, 35, v1

// SICI: s_mov_b64 s[0:1], 0x4d2 ; encoding: [0xff,0x04,0x80,0xbe,0xd2,0x04,0x00,0x00]
// GFX89: s_mov_b64 s[0:1], 0x4d2 ; encoding: [0xff,0x01,0x80,0xbe,0xd2,0x04,0x00,0x00]
s_mov_b64_e32 s[0:1], 1234

// SICI: v_and_b32_e32 v0, 0x4d2, v1 ; encoding: [0xff,0x02,0x00,0x36,0xd2,0x04,0x00,0x00]
// GFX89: v_and_b32_e32 v0, 0x4d2, v1 ; encoding: [0xff,0x02,0x00,0x26,0xd2,0x04,0x00,0x00]
v_and_b32_e32 v0, 1234, v1

// NOSICI: error: invalid literal operand
// NOGFX89: error: invalid literal operand
v_and_b32_e64 v0, 1234, v1

// SICI: s_mov_b64 s[0:1], 0xffff2bcf ; encoding: [0xff,0x04,0x80,0xbe,0xcf,0x2b,0xff,0xff]
// GFX89: s_mov_b64 s[0:1], 0xffff2bcf ; encoding: [0xff,0x01,0x80,0xbe,0xcf,0x2b,0xff,0xff]
s_mov_b64_e32 s[0:1], -54321

// SICI: v_and_b32_e32 v0, 0xffff2bcf, v1 ; encoding: [0xff,0x02,0x00,0x36,0xcf,0x2b,0xff,0xff]
// GFX89: v_and_b32_e32 v0, 0xffff2bcf, v1 ; encoding: [0xff,0x02,0x00,0x26,0xcf,0x2b,0xff,0xff]
v_and_b32_e32 v0, -54321, v1

// SICI: s_mov_b64 s[0:1], 0xdeadbeef ; encoding: [0xff,0x04,0x80,0xbe,0xef,0xbe,0xad,0xde]
// GFX89: s_mov_b64 s[0:1], 0xdeadbeef ; encoding: [0xff,0x01,0x80,0xbe,0xef,0xbe,0xad,0xde]
s_mov_b64_e32 s[0:1], 0xdeadbeef

// SICI: v_and_b32_e32 v0, 0xdeadbeef, v1 ; encoding: [0xff,0x02,0x00,0x36,0xef,0xbe,0xad,0xde]
// GFX89: v_and_b32_e32 v0, 0xdeadbeef, v1 ; encoding: [0xff,0x02,0x00,0x26,0xef,0xbe,0xad,0xde]
v_and_b32_e32 v0, 0xdeadbeef, v1

// SICI: s_mov_b64 s[0:1], 0xffffffff ; encoding: [0xff,0x04,0x80,0xbe,0xff,0xff,0xff,0xff]
// GFX89: s_mov_b64 s[0:1], 0xffffffff ; encoding: [0xff,0x01,0x80,0xbe,0xff,0xff,0xff,0xff]
s_mov_b64_e32 s[0:1], 0xffffffff

// SICI: v_and_b32_e32 v0, -1, v1 ; encoding: [0xc1,0x02,0x00,0x36]
// GFX89: v_and_b32_e32 v0, -1, v1 ; encoding: [0xc1,0x02,0x00,0x26]
v_and_b32_e32 v0, 0xffffffff, v1

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
s_mov_b64_e32 s[0:1], 0x123456789abcdef0

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
v_and_b32_e32 v0, 0x123456789abcdef0, v1

// SICI: s_mov_b64 s[0:1], -1 ; encoding: [0xc1,0x04,0x80,0xbe]
// GFX89: s_mov_b64 s[0:1], -1 ; encoding: [0xc1,0x01,0x80,0xbe]
s_mov_b64_e32 s[0:1], 0xffffffffffffffff

// SICI: v_and_b32_e32 v0, -1, v1 ; encoding: [0xc1,0x02,0x00,0x36]
// GFX89: v_and_b32_e32 v0, -1, v1 ; encoding: [0xc1,0x02,0x00,0x26]
v_and_b32_e32 v0, 0xffffffffffffffff, v1

//---------------------------------------------------------------------------//
// 1/(2*PI)
//---------------------------------------------------------------------------//

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
v_trunc_f32_e32 v0, 0x3fc45f306dc9c882

// NOSICI: error: invalid operand for instruction
// GFX89: v_fract_f64_e32 v[0:1], 0.15915494309189532 ; encoding: [0xf8,0x64,0x00,0x7e]
v_fract_f64_e32 v[0:1], 0x3fc45f306dc9c882

// SICI: v_trunc_f32_e32 v0, 0x3e22f983 ; encoding: [0xff,0x42,0x00,0x7e,0x83,0xf9,0x22,0x3e]
// GFX89: v_trunc_f32_e32 v0, 0.15915494 ; encoding: [0xf8,0x38,0x00,0x7e]
v_trunc_f32_e32 v0, 0x3e22f983

// SICI: v_fract_f64_e32 v[0:1], 0x3e22f983 ; encoding: [0xff,0x7c,0x00,0x7e,0x83,0xf9,0x22,0x3e]
// GFX89: v_fract_f64_e32 v[0:1], 0x3e22f983 ; encoding: [0xff,0x64,0x00,0x7e,0x83,0xf9,0x22,0x3e]
v_fract_f64_e32 v[0:1], 0x3e22f983

// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction
v_trunc_f32_e64 v0, 0x3fc45f306dc9c882

// NOSICI: error: invalid operand for instruction
// GFX89: v_fract_f64_e64 v[0:1], 0.15915494309189532 ; encoding: [0x00,0x00,0x72,0xd1,0xf8,0x00,0x00,0x00]
v_fract_f64_e64 v[0:1], 0x3fc45f306dc9c882

// NOSICI: error: invalid literal operand
// GFX89: v_trunc_f32_e64 v0, 0.15915494 ; encoding: [0x00,0x00,0x5c,0xd1,0xf8,0x00,0x00,0x00]
v_trunc_f32_e64 v0, 0x3e22f983

// NOSICI: error: invalid literal operand
// NOGFX89: error: invalid literal operand
v_fract_f64_e64 v[0:1], 0x3e22f983

// NOSICI: error: invalid operand for instruction
// GFX89: s_mov_b64 s[0:1], 0.15915494309189532 ; encoding: [0xf8,0x01,0x80,0xbe]
s_mov_b64_e32 s[0:1], 0.159154943091895317852646485335

// SICI: v_and_b32_e32 v0, 0x3e22f983, v1 ; encoding: [0xff,0x02,0x00,0x36,0x83,0xf9,0x22,0x3e]
// GFX89: v_and_b32_e32 v0, 0.15915494, v1 ; encoding: [0xf8,0x02,0x00,0x26]
v_and_b32_e32 v0, 0.159154943091895317852646485335, v1

// NOSICI: error: invalid literal operand
// GFX89: v_and_b32_e64 v0, 0.15915494, v1 ; encoding: [0x00,0x00,0x13,0xd1,0xf8,0x02,0x02,0x00]
v_and_b32_e64 v0, 0.159154943091895317852646485335, v1

// SICI: v_fract_f64_e32 v[0:1], 0x3fc45f30 ; encoding: [0xff,0x7c,0x00,0x7e,0x30,0x5f,0xc4,0x3f]
// GFX89: v_fract_f64_e32 v[0:1], 0.15915494309189532 ; encoding: [0xf8,0x64,0x00,0x7e]
v_fract_f64 v[0:1], 0.159154943091895317852646485335

// SICI: v_trunc_f32_e32 v0, 0x3e22f983 ; encoding: [0xff,0x42,0x00,0x7e,0x83,0xf9,0x22,0x3e]
// GFX89: v_trunc_f32_e32 v0, 0.15915494 ; encoding: [0xf8,0x38,0x00,0x7e]
v_trunc_f32 v0, 0.159154943091895317852646485335

//---------------------------------------------------------------------------//
// integer literal truncation checks
//---------------------------------------------------------------------------//

// NOGCN: error: invalid operand for instruction
s_mov_b32 s0, 0x101ffffffff

// NOGCN: error: invalid operand for instruction
s_mov_b32 s0, 0x1000000001

// NOGCN: error: invalid operand for instruction
s_mov_b32 s0, 0x1000000fff

// NOGCN: error: invalid operand for instruction
v_trunc_f32 v0, 0x1fffffffff0

// NOGCN: error: invalid operand for instruction
v_trunc_f32 v0, 0x100000001

// NOGCN: error: invalid operand for instruction
v_trunc_f32 v0, 0x1fffffff000

// NOGCN: error: invalid operand for instruction
s_mov_b64 s[0:1], 0x101ffffffff

// NOGCN: error: invalid operand for instruction
s_mov_b64 s[0:1], 0x1000000001

// NOGCN: error: invalid operand for instruction
s_mov_b64 s[0:1], 0x1000000fff

// NOGCN: error: invalid operand for instruction
v_trunc_f64 v[0:1], 0x1fffffffff0

// NOGCN: error: invalid operand for instruction
v_trunc_f64 v[0:1], 0x100000001

// NOGCN: error: invalid operand for instruction
v_trunc_f64 v[0:1], 0x1fffffff000

//---------------------------------------------------------------------------//
// named inline values like shared_base
//---------------------------------------------------------------------------//

// NOSICIVI: error: not a valid operand.
// GFX9: buffer_atomic_add v0, off, s[0:3], src_shared_base offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0x00,0x00,0xeb]
buffer_atomic_add v0, off, s[0:3], src_shared_base offset:4095

// NOSICIVI: error: not a valid operand.
// GFX9: s_add_i32 s0, src_shared_base, s0 ; encoding: [0xeb,0x00,0x00,0x81]
s_add_i32 s0, src_shared_base, s0

// NOSICIVI: error: not a valid operand.
// GFX9: s_add_i32 s0, src_shared_limit, s0 ; encoding: [0xec,0x00,0x00,0x81]
s_add_i32 s0, src_shared_limit, s0

// NOSICIVI: error: not a valid operand.
// GFX9: s_add_i32 s0, src_private_base, s0 ; encoding: [0xed,0x00,0x00,0x81]
s_add_i32 s0, src_private_base, s0

// NOSICIVI: error: not a valid operand.
// GFX9: s_add_i32 s0, src_private_limit, s0 ; encoding: [0xee,0x00,0x00,0x81]
s_add_i32 s0, src_private_limit, s0

// NOSICIVI: error: not a valid operand.
// GFX9: s_add_i32 s0, src_pops_exiting_wave_id, s0 ; encoding: [0xef,0x00,0x00,0x81]
s_add_i32 s0, src_pops_exiting_wave_id, s0

// NOSICIVI: error: not a valid operand.
// GFX9: s_and_b64 s[0:1], s[0:1], src_shared_base ; encoding: [0x00,0xeb,0x80,0x86]
s_and_b64 s[0:1], s[0:1], src_shared_base

// NOSICIVI: error: not a valid operand.
// GFX9: s_and_b64 s[0:1], s[0:1], src_shared_limit ; encoding: [0x00,0xec,0x80,0x86]
s_and_b64 s[0:1], s[0:1], src_shared_limit

// NOSICIVI: error: not a valid operand.
// GFX9: s_and_b64 s[0:1], s[0:1], src_private_base ; encoding: [0x00,0xed,0x80,0x86]
s_and_b64 s[0:1], s[0:1], src_private_base

// NOSICIVI: error: not a valid operand.
// GFX9: s_and_b64 s[0:1], s[0:1], src_private_limit ; encoding: [0x00,0xee,0x80,0x86]
s_and_b64 s[0:1], s[0:1], src_private_limit

// NOSICIVI: error: not a valid operand.
// GFX9: s_and_b64 s[0:1], s[0:1], src_pops_exiting_wave_id ; encoding: [0x00,0xef,0x80,0x86]
s_and_b64 s[0:1], s[0:1], src_pops_exiting_wave_id

// NOSICIVI: error: not a valid operand.
// GFX9: v_add_u16_e32 v0, src_shared_base, v0 ; encoding: [0xeb,0x00,0x00,0x4c]
v_add_u16 v0, src_shared_base, v0

// NOSICIVI: error: not a valid operand.
// GFX9: v_add_u16_sdwa v0, src_shared_base, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x00,0x00,0x4c,0xeb,0x06,0x86,0x06]
v_add_u16_sdwa v0, src_shared_base, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

// NOSICIVI: error: not a valid operand.
// GFX9: v_add_u16_sdwa v0, v0, src_shared_base dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0xd6,0x01,0x4c,0x00,0x06,0x06,0x86]
v_add_u16_sdwa v0, v0, src_shared_base dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

// NOSICIVI: error: not a valid operand.
// GFX9: v_add_u32_e32 v0, src_shared_base, v0 ; encoding: [0xeb,0x00,0x00,0x68]
v_add_u32 v0, src_shared_base, v0

// NOSICIVI: error: not a valid operand.
// GFX9: v_add_u32_e64 v0, src_shared_base, v0 ; encoding: [0x00,0x00,0x34,0xd1,0xeb,0x00,0x02,0x00]
v_add_u32_e64 v0, src_shared_base, v0

// NOSICIVI: error: not a valid operand.
// GFX9: v_cmp_eq_i64_e32 vcc, src_shared_base, v[0:1] ; encoding: [0xeb,0x00,0xc4,0x7d]
v_cmp_eq_i64 vcc, src_shared_base, v[0:1]

// NOSICIVI: error: not a valid operand.
// GFX9: v_max_f16_e32 v0, src_shared_base, v0 ; encoding: [0xeb,0x00,0x00,0x5a]
v_max_f16 v0, src_shared_base, v0

// NOSICIVI: error: not a valid operand.
// GFX9: v_max_f32_e32 v0, src_shared_base, v0 ; encoding: [0xeb,0x00,0x00,0x16]
v_max_f32 v0, src_shared_base, v0

// NOSICIVI: error: not a valid operand.
// GFX9: v_max_f64 v[0:1], src_shared_base, v[0:1] ; encoding: [0x00,0x00,0x83,0xd2,0xeb,0x00,0x02,0x00]
v_max_f64 v[0:1], src_shared_base, v[0:1]

// NOSICIVI: error: not a valid operand.
// GFX9: v_pk_add_f16 v0, src_shared_base, v0 ; encoding: [0x00,0x00,0x8f,0xd3,0xeb,0x00,0x02,0x18]
v_pk_add_f16 v0, src_shared_base, v0

// NOSICI: error: invalid operand for instruction
// NOVI: error: not a valid operand.
// GFX9: v_ceil_f16_e64 v0, -src_shared_base ; encoding: [0x00,0x00,0x85,0xd1,0xeb,0x00,0x00,0x20]
v_ceil_f16 v0, neg(src_shared_base)

// NOSICI: error: invalid operand for instruction
// NOVI: error: not a valid operand.
// GFX9: v_ceil_f16_e64 v0, |src_shared_base| ; encoding: [0x00,0x01,0x85,0xd1,0xeb,0x00,0x00,0x00]
v_ceil_f16 v0, abs(src_shared_base)

// NOSICIVI: error: not a valid operand.
// GFX9: v_ceil_f64_e64 v[5:6], |src_shared_base| ; encoding: [0x05,0x01,0x58,0xd1,0xeb,0x00,0x00,0x00]
v_ceil_f64 v[5:6], |src_shared_base|

// NOSI: error: invalid operand for instruction
// NOCIVI: error: not a valid operand.
// GFX9: v_ceil_f64_e64 v[5:6], -src_shared_base ; encoding: [0x05,0x00,0x58,0xd1,0xeb,0x00,0x00,0x20]
v_ceil_f64 v[5:6], -src_shared_base

// NOSICIVI: error: not a valid operand.
// GFX9: v_ceil_f32_e64 v0, -src_shared_base ; encoding: [0x00,0x00,0x5d,0xd1,0xeb,0x00,0x00,0x20]
v_ceil_f32 v0, -src_shared_base

// NOSICIVI: error: not a valid operand.
// GFX9: v_ceil_f32_e64 v0, |src_shared_base| ; encoding: [0x00,0x01,0x5d,0xd1,0xeb,0x00,0x00,0x00]
v_ceil_f32 v0, |src_shared_base|

// NOSICIVI: error: not a valid operand.
// GFX9: v_ceil_f16_sdwa v5, |src_shared_base| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x8a,0x0a,0x7e,0xeb,0x16,0xa6,0x00]
v_ceil_f16_sdwa v5, |src_shared_base| dst_sel:DWORD dst_unused:UNUSED_PRESERVE

// NOSICIVI: error: not a valid operand.
// GFX9: v_ceil_f16_sdwa v5, -src_shared_base dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x8a,0x0a,0x7e,0xeb,0x16,0x96,0x00]
v_ceil_f16_sdwa v5, -src_shared_base dst_sel:DWORD dst_unused:UNUSED_PRESERVE

// NOSICIVI: error: not a valid operand.
// GFX9: v_ceil_f32_sdwa v5, src_shared_base dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x3a,0x0a,0x7e,0xeb,0x16,0x86,0x00]
v_ceil_f32_sdwa v5, src_shared_base dst_sel:DWORD src0_sel:DWORD

// NOSICIVI: error: not a valid operand.
// GFX9: v_ceil_f32_sdwa v5, |src_shared_base| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x3a,0x0a,0x7e,0xeb,0x16,0xa6,0x00]
v_ceil_f32_sdwa v5, |src_shared_base| dst_sel:DWORD src0_sel:DWORD

//---------------------------------------------------------------------------//
// named inline values compete with other scalars for constant bus access
//---------------------------------------------------------------------------//

// NOSICIVI: error: not a valid operand.
// NOGFX9: error: invalid operand (violates constant bus restrictions)
v_add_u32 v0, private_base, s0

// v_div_fmas implicitly reads VCC
// NOSICIVI: error: not a valid operand.
// NOGFX9: error: invalid operand (violates constant bus restrictions)
v_div_fmas_f32 v0, shared_base, v0, v1

// v_div_fmas implicitly reads VCC
// NOSICIVI: error: not a valid operand.
// NOGFX9: error: invalid operand (violates constant bus restrictions)
v_div_fmas_f32 v0, v0, shared_limit, v1

// v_div_fmas implicitly reads VCC
// NOSICIVI: error: not a valid operand.
// NOGFX9: error: invalid operand (violates constant bus restrictions)
v_div_fmas_f32 v0, v0, v1, private_limit

// v_addc_co_u32 implicitly reads VCC (VOP2)
// NOSICIVI: error: not a valid operand.
// NOGFX9: error: invalid operand (violates constant bus restrictions)
v_addc_co_u32 v0, vcc, shared_base, v0, vcc

// NOSICIVI: error: not a valid operand.
// NOGFX9: error: invalid operand (violates constant bus restrictions)
v_madak_f32 v0, shared_base, v0, 0x11213141

// NOSICIVI: error: not a valid operand.
// NOGFX9: error: invalid operand (violates constant bus restrictions)
v_cmp_eq_f32 s[0:1], private_base, private_limit

// NOSICIVI: error: not a valid operand.
// NOGFX9: error: invalid operand (violates constant bus restrictions)
v_cmp_eq_f32 s[0:1], private_base, s0

// NOSICIVI: error: not a valid operand.
// NOGFX9: error: invalid operand (violates constant bus restrictions)
v_pk_add_f16 v255, private_base, private_limit
