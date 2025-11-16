// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --implicit-check-not=error: %s

ds_gws_barrier v1 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_barrier v1 offset:65535 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_init v1 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_init v1 offset:65535 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_br v1 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_br v1 offset:65535 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_p nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_p offset:65535 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_release_all nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_release_all offset:65535 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_v nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed

ds_gws_sema_v offset:65535 nogds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: nogds is not allowed
