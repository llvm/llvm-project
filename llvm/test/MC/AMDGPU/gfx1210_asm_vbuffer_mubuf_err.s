// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

buffer_load_b32 v5, v1, s[8:11], s3 offen offset:4095 scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}buffer_load_b32 v5, v1, s[8:11], s3 offen offset:4095 scale_offset
// GFX1210-ERR-NEXT:{{^}}                                                      ^

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
