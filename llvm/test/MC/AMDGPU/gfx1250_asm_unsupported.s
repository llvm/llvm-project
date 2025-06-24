; RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

;; All "TBUFFER" ops, and BUFFER_LOAD/STORE_FORMAT ops.

tbuffer_load_d16_format_x v4, off, s[8:11], s3 format:[BUF_FMT_8_UNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_SINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_8_8_USCALED] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_32_SINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_SSCALED] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, s[8:11], s3 format:[BUF_FMT_11_11_10_FLOAT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_8_8_8_UINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, s[8:11], s3 format:[BUF_FMT_32_32_32_32_UINT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, s[8:11], s3 format:[BUF_FMT_32_FLOAT] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SNORM] offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_xy v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_xyz v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_xyzw v[5:6], off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_xy v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_xyz v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_xyzw v[1:2], off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_d16_hi_x v1, off, s[12:15], s4 offset:8388607
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
