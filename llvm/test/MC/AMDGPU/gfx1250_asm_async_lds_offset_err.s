// RUN: not llvm-mc -triple=amdgpu12.50 -filetype=null %s 2>&1 | FileCheck --implicit-check-not=error: %s

global_load_async_to_lds_b8 v1, v[2:3], off offset:-64
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

global_load_async_to_lds_b32 v1, v2, s[2:3] offset:-64
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

global_load_async_to_lds_b64 v1, v[2:3], off offset:-1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

global_load_async_to_lds_b128 v1, v2, s[2:3] offset:0x10000
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

global_load_async_to_lds_b128 v2, v2, s[4:5] offset:0x10100
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

global_load_async_to_lds_b32 v1, v[2:3], off offset:-8388608
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

cluster_load_async_to_lds_b8 v1, v2, s[2:3] offset:-64
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

cluster_load_async_to_lds_b32 v1, v[2:3], off offset:-64
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

cluster_load_async_to_lds_b64 v1, v2, s[2:3] offset:0x10100
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

cluster_load_async_to_lds_b128 v1, v[2:3], off offset:8388607
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset for an async load to LDS

global_load_async_to_lds_b128 v1, v[2:3], off offset:0xffff
global_load_async_to_lds_b32 v1, v2, s[2:3] offset:0xffc0
cluster_load_async_to_lds_b64 v1, v[2:3], off
global_store_async_from_lds_b128 v[2:3], v1, off offset:-64
global_store_async_from_lds_b32 v2, v1, s[2:3] offset:0x10100
