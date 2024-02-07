// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx950 %s 2>&1 | FileCheck -check-prefix=ERR %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx950 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck -check-prefix=W32-ERR %s

//===----------------------------------------------------------------------===//
// v_mfma_f32_32x32x4_xf32
//===----------------------------------------------------------------------===//

v_mfma_f32_32x32x4_xf32  a[0:3], v[2:3], v[4:5], a[2:5]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], v[0:3], v[0:3], v[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], v[0:3], v[0:3], v[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], a[0:3], v[0:3], 1.0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], v[0:3], a[0:3], 1.0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], v[0:3], v[0:3], v[0:3] blgp:5
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3] abid:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], v[0:3], v[0:3], a[4:7]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], a[0:3], a[0:3], v[4:7]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], v[2:3], v[4:5], a[2:5]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], v[0:3], v[0:3], v[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], v[0:3], v[0:3], v[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], a[0:3], v[0:3], 1.0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], v[0:3], a[0:3], 1.0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], v[0:3], v[0:3], v[0:3] blgp:5
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3] abid:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  a[0:3], v[0:3], v[0:3], a[4:7]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32  v[0:3], a[0:3], a[0:3], v[4:7]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU


//===----------------------------------------------------------------------===//
// v_mfma_f32_16x16x8_xf32
//===----------------------------------------------------------------------===//

v_mfma_f32_16x16x8_xf32 a[0:3], v[2:3], v[4:5], a[2:5]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], v[0:3], v[0:3], v[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], v[0:3], v[0:3], v[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], a[0:3], v[0:3], 1.0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], v[0:3], a[0:3], 1.0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], v[0:3], v[0:3], v[0:3] blgp:5
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3] abid:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], v[0:3], v[0:3], a[4:7]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], a[0:3], a[0:3], v[4:7]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU


v_mfma_f32_16x16x8_xf32 a[0:3], v[2:3], v[4:5], a[2:5]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], v[0:3], v[0:3], v[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], v[0:3], v[0:3], v[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], a[0:3], v[0:3], 1.0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], v[0:3], a[0:3], 1.0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], v[0:3], v[0:3], v[0:3] blgp:5
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3] blgp:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3] abid:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], a[0:3], a[0:3], a[0:3] cbsz:3 abid:1
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 a[0:3], v[0:3], v[0:3], a[4:7]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8_xf32 v[0:3], a[0:3], a[0:3], v[4:7]
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// ds_read_b64_tr_b4
//===----------------------------------------------------------------------===//
ds_read_b64_tr_b4 v[1:2], v0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b64_tr_b4 v1, v0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b64_tr_b4 v[0:1], s0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b64_tr_b4 v[2:3], v2 offset:-64
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
//ds_read_b64_tr_b8
//===----------------------------------------------------------------------===//
ds_read_b64_tr_b8 v[1:2], v0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b64_tr_b8 v1, v0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b64_tr_b8 v[0:1], s0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b64_tr_b8 v[2:3], v2 offset:-64
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// ds_read_b64_tr_b16
//===----------------------------------------------------------------------===//
ds_read_b64_tr_b16 v[1:2], v0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b64_tr_b16 v1, v0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b64_tr_b16 v[0:1], s0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b64_tr_b16 v[2:3], v2 offset:-64
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// ds_read_b96_tr_b6
//===----------------------------------------------------------------------===//
ds_read_b96_tr_b6 v[1:3], v0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b96_tr_b6 v1, v0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b96_tr_b6 v[0:3], s0
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_read_b96_tr_b6 v[2:4], v2 offset:-64
// ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
