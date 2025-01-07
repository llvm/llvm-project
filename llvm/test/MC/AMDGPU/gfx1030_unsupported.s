// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize32 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_mul_lo_i32 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// Unsupported dpp variants.
//===----------------------------------------------------------------------===//

v_fmac_legacy_f32_dpp v255, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

//===----------------------------------------------------------------------===//
// Unsupported sdwa variants.
//===----------------------------------------------------------------------===//

v_fmac_legacy_f32_sdwa v255, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
