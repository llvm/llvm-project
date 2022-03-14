// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck -check-prefix=GFX940 %s

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[0:31] neg:[1,0,0]
// GFX940: error: invalid modifier: neg is not supported

v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[0:31] neg:[1,0,0]
// GFX940: error: invalid modifier: neg is not supported

v_mfma_f64_16x16x4_f64 v[0:7], v[0:1], v[2:3], v[0:7] blgp:7
// GFX940: error: invalid modifier: blgp is not supported

v_mfma_f64_16x16x4f64 v[0:7], v[0:1], v[2:3], v[0:7] blgp:7
// GFX940: error: invalid modifier: blgp is not supported

v_mfma_f64_16x16x4_f64 a[0:7], v[0:1], v[2:3], a[0:7] blgp:7
// GFX940: error: invalid modifier: blgp is not supported

v_mfma_f64_4x4x4_4b_f64 v[0:1], v[0:1], a[2:3], v[2:3] blgp:7
// GFX940: error: invalid modifier: blgp is not supported

v_mfma_f64_4x4x4_4b_f64 a[0:1], v[0:1], a[2:3], a[2:3] blgp:7
// GFX940: error: invalid modifier: blgp is not supported
