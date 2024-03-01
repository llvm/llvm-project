// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

buffer_load_format_d16_x v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_xy v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_xyz v[5:6], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_xyzw v[5:6], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_x v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_xy v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_xyz v[1:2], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_xyzw v[1:2], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
