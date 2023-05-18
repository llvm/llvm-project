// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck -check-prefix=GFX940 %s

v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, 0
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x2bf16 a[0:15], v0, v1, 0
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_4x4x2bf16 a[0:3], v0, v1, 0
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4bf16 a[0:15], v0, v1, 0
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

v_mfma_f32_16x16x8bf16 a[0:3], v0, v1, 0
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[0:31] neg:[1,0,0]
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid modifier: neg is not supported

v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[0:31] neg:[1,0,0]
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid modifier: neg is not supported

v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7] blgp:7
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid modifier: blgp is not supported

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[0:7] blgp:7
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid modifier: blgp is not supported

v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7] blgp:7
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid modifier: blgp is not supported

v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], a[2:3], v[2:3] blgp:7
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid modifier: blgp is not supported

v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] blgp:7
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: invalid modifier: blgp is not supported

v_mfma_i32_32x32x8i8 a[0:15], v0, v1, a[0:15]
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_i32_16x16x16i8 a[0:3], v0, v1, a[0:3]
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[2:9]
// GFX940: :[[@LINE-1]]:{{[0-9]+}}: error: source 2 operand must not partially overlap with dst
