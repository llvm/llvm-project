// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck %s --check-prefix=NOGFX940 --implicit-check-not=error:

image_load v[4:6], v[238:241], s[28:35] dmask:0x7 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_load_pck v5, v[0:3], s[8:15] dmask:0x1 glc
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_load_pck_sgn v5, v[0:3], s[8:15] dmask:0x1 lwe
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_load_mip v5, v[0:3], s[8:15]
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_load_mip_pck v5, v1, s[8:15] dmask:0x1
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_load_mip_pck_sgn v[4:5], v[0:3], s[8:15] dmask:0x5
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_store v[192:194], v[238:241], s[28:35] dmask:0x7 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_store_pck v1, v[2:5], s[12:19] dmask:0x1 unorm da
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_store_mip v1, v[2:5], s[12:19]
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_store_mip_pck v252, v[2:3], s[12:19] dmask:0x1 a16
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_add v4, v192, s[28:35] dmask:0x1 unorm glc
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_and v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_swap v4, v[192:195], s[28:35] dmask:0x1 unorm glc
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_cmpswap v[4:5], v[192:195], s[28:35] dmask:0x3 unorm glc
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_or v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_xor v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_sub v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_smin v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_smax v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_umin v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_umax v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_inc v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_atomic_dec v4, v192, s[28:35] dmask:0x1 unorm
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_get_resinfo v5, v1, s[8:15] dmask:0x1
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_sample v5, v[0:3], s[8:15], s[12:15] dmask:0x1
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:

image_gather4 v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x2
// NOGFX940: :[[@LINE-1]]:{{[0-9]+}}: error:
