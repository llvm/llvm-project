// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Unsupported instructions.
//===----------------------------------------------------------------------===//

s_waitcnt_expcnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_waitcnt_lgkmcnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_waitcnt_vmcnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_waitcnt_vscnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_subvector_loop_begin s0, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_subvector_loop_end s0, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_cdbgsys 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_cdbguser 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_cdbgsys_or_user 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_cbranch_cdbgsys_and_user 0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_legacy_f32 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dot2c_f32_f16 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_max_f32 v0, v1, v2 :: v_dual_max_f32 v3, v4, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_dual_min_f32 v0, v1, v2 :: v_dual_min_f32 v3, v4, v5
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
