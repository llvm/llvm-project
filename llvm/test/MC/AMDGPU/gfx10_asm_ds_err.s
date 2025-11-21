// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

ds_gws_sema_release_all nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_release_all offset:4660 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_init v0 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_init v0 offset:0 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_v nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_v offset:65535 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_br v0 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_br v0 offset:4660 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_p nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_p offset:0 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_barrier v0 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_barrier v0 offset:0 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed
