// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx9-4-generic -mattr=+wavefrontsize32 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx9-4-generic -mattr=+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_mfma_f32_16x16x8_xf32 a[0:3], v[2:3], v[4:5], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x8xf32 a[0:3], v[2:3], v[4:5], a[2:5]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4_xf32 a[0:15], v[2:3], v[4:5], a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x4xf32 a[0:15], v[2:3], v[4:5], a[18:33]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8 v1, 3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf8_sdwa v5, v1 src0_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8 v1, 3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8_dpp v5, v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8_e64 v5, v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_fp8_sdwa v5, v1 src0_sel:BYTE_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_fp8 v[0:1], v3 quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_fp8_dpp v[10:11], v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_fp8_sdwa v[10:11], v1 src0_sel:WORD_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_bf8 v[0:1], v3 quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_bf8_dpp v[10:11], v1 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_pk_f32_bf8_sdwa v[10:11], v1 src0_sel:WORD_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_bf8 a[0:3], v[2:3], v[4:5], a[0:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_bf8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[2:3], v[4:5], a[0:3]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_bf8 a[0:15], v[2:3], v[4:5], a[0:15]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_bf8_fp8 a[0:15], v[2:3], v[4:5], a[0:15]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_bf8 a[0:15], v[2:3], v[4:5], a[0:15]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mfma_f32_32x32x16_fp8_fp8 a[0:15], v[2:3], v[4:5], a[0:15]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64_bf8_bf8 a[0:3], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64_bf8_fp8 a[0:3], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64_fp8_bf8 a[0:3], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_16x16x64_fp8_fp8 a[0:3], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32_bf8_bf8 a[0:15], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32_bf8_fp8 a[0:15], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32_fp8_bf8 a[0:15], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_smfmac_f32_32x32x32_fp8_fp8 a[0:15], v[2:3], a[4:7], v1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
