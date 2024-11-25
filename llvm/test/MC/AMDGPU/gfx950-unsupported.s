// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx950 %s 2>&1 | FileCheck -check-prefix=ERR %s

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
