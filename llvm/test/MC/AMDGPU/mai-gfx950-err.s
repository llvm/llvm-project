// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx950 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_mfma_ld_scale_b32 v0, 65
// CHECK: :[[@LINE-1]]:25: error: literal operands are not supported

v_mfma_ld_scale_b32 65, v0
// CHECK: :[[@LINE-1]]:21: error: literal operands are not supported

v_mfma_ld_scale_b32 65, 65
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported
// CHECK-NEXT:{{^}}v_mfma_ld_scale_b32 65, 65
// CHECK-NEXT:{{^}}                    ^

v_mfma_ld_scale_b32 s0, s1
// CHECK: :[[@LINE-1]]:25: error: invalid operand (violates constant bus restrictions)

v_mfma_ld_scale_b32 v0, v0 clamp
// CHECK: :[[@LINE-1]]:28: error: invalid operand for instruction

v_mfma_ld_scale_b32 v0, v0 neg_lo:[0,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand

v_mfma_ld_scale_b32 v0, v0 neg_lo:[1,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand

v_mfma_ld_scale_b32 v0, v0 neg_hi:[1,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand

v_mfma_ld_scale_b32 v0, v0 neg_hi:[0,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand

v_mfma_ld_scale_b32 v0, v0 neg_lo:[0,1] neg_hi:[0,1]
// CHECK: :[[@LINE-1]]:28: error: not a valid operand


v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3] cbsz:2
// CHECK: :[[@LINE-1]]:37: error: wrong register tuple size for cbsz value 2

v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3] blgp:2
// CHECK: :[[@LINE-1]]:46: error: wrong register tuple size for blgp value 2

v_mfma_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3] cbsz:2 blgp:2
// CHECK: :[[@LINE-1]]:37: error: wrong register tuple size for cbsz value 2
// CHECK: :[[@LINE-2]]:46: error: wrong register tuple size for blgp value 2


v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[0:3] cbsz:2
// CHECK: :[[@LINE-1]]:37: error: wrong register tuple size for cbsz value 2

v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[0:3] blgp:2
// CHECK: :[[@LINE-1]]:46: error: wrong register tuple size for blgp value 2

v_mfma_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[0:3] cbsz:2 blgp:2
// CHECK: :[[@LINE-1]]:37: error: wrong register tuple size for cbsz value 2
// CHECK: :[[@LINE-2]]:46: error: wrong register tuple size for blgp value 2

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3] v20, v21 cbsz:2
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 2

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[0:3] v20, v21 cbsz:2
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 2

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3] v20, v21 blgp:2
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 2

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:19], a[0:3] v20, v21 blgp:2
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 2

v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[0:15] cbsz:2
// CHECK: :[[@LINE-1]]:37: error: wrong register tuple size for cbsz value 2

v_mfma_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[0:15] blgp:2
// CHECK: :[[@LINE-1]]:47: error: wrong register tuple size for blgp value 2

v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[0:15] cbsz:2
// CHECK: :[[@LINE-1]]:37: error: wrong register tuple size for cbsz value 2

v_mfma_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[0:15] blgp:2
// CHECK: :[[@LINE-1]]:47: error: wrong register tuple size for blgp value 2

v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[0:15] v32, v33 cbsz:2
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 2

v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[0:15] v32, v33 cbsz:2
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 2



v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:19], v[0:3] v20, v21
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 0

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:19], v[0:3] v20, v21 cbsz:1
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 1

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:19], v[0:3] v20, v21 cbsz:2
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 2

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:19], v[0:3] v20, v21 cbsz:3
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 3


v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[0:3] v20, v21
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 0

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[0:3] v20, v21 blgp:1
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 1

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[0:3] v20, v21 blgp:2
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 2

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:15], v[0:3] v20, v21 blgp:3
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 3

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:19], a[0:3] v20, v21
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 0

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:19], a[0:3] v20, v21 cbsz:1
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 1

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:19], a[0:3] v20, v21 cbsz:2
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 2

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:7], a[12:19], a[0:3] v20, v21 cbsz:3
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 3

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:15], a[0:3] v20, v21
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 0

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:15], a[0:3] v20, v21 blgp:1
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 1

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:15], a[0:3] v20, v21 blgp:2
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 2

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[4:11], a[12:15], a[0:3] v20, v21 blgp:3
// CHECK: :[[@LINE-1]]:52: error: wrong register tuple size for blgp value 3

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[12:19], v[0:3] v20, v21 cbsz:3
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 3

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3] v20, v21 cbsz:3
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 3

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[12:19], v[4:7], v[0:3] v20, v21 blgp:3
// CHECK: :[[@LINE-1]]:53: error: wrong register tuple size for blgp value 3

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[12:19], v[4:11], v[0:3] v20, v21 blgp:3
// CHECK: :[[@LINE-1]]:53: error: wrong register tuple size for blgp value 3

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:9], v[12:19], v[0:3] v20, v21 cbsz:4
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 4

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[0:3] v20, v21 cbsz:4
// CHECK: :[[@LINE-1]]:43: error: wrong register tuple size for cbsz value 4

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[12:19], v[4:9], v[0:3] v20, v21 blgp:4
// CHECK: :[[@LINE-1]]:53: error: wrong register tuple size for blgp value 4

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[12:19], v[4:11], v[0:3] v20, v21 blgp:4
// CHECK: :[[@LINE-1]]:53: error: wrong register tuple size for blgp value 4


// Workaround a hardware bug to disallow sgpr/inline constants as scale operands

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v44, s24
// CHECK: :[[@LINE-1]]:77: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], s24, v44
// CHECK: :[[@LINE-1]]:72: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], m0, v24
// CHECK: :[[@LINE-1]]:72: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], vcc_lo, v24
// CHECK: :[[@LINE-1]]:72: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], 9, v24
// CHECK: :[[@LINE-1]]:72: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, 9
// CHECK: :[[@LINE-1]]:77: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], 33, v24
// CHECK: :[[@LINE-1]]:72: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], 4.0, v24
// CHECK: :[[@LINE-1]]:72: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], v24, 4.0
// CHECK: :[[@LINE-1]]:77: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], -4.0, v24
// CHECK: :[[@LINE-1]]:72: error: invalid operand for instruction

v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], v[20:23], 0.15915494, v24
// CHECK: :[[@LINE-1]]:72: error: invalid operand for instruction

v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], 16, v49
// CHECK: :[[@LINE-1]]:73: error: invalid operand for instruction

v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], v48, -4.0
// CHECK: :[[@LINE-1]]:78: error: invalid operand for instruction

v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], 4.0, v24
// CHECK: :[[@LINE-1]]:73: error: invalid operand for instruction

v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:23], v[24:31], v[32:47], 0.15915494, v24
// CHECK: :[[@LINE-1]]:73: error: invalid operand for instruction
