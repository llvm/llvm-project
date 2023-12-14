// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Unsupported instructions.
//===----------------------------------------------------------------------===//

s_subvector_loop_begin s0, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_subvector_loop_end s0, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_f32 v0, v1, v2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_rtn_f32 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_f64 v0, v[1:2], v[3:4]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_cmpstore_rtn_f64 v[0:1], v2, v[3:4], v[5:6]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_add_gs_reg_rtn v[0:1], v2 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_sub_gs_reg_rtn v[0:1], v2 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_wrap_rtn_b32 v0, v1, v2, v3
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_sema_release_all gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_init v0 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_sema_v gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_sema_br v0 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_sema_p gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_gws_barrier v0 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_ordered_count v0, v1 gds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
