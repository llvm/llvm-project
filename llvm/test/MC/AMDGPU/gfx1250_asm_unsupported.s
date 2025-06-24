; RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

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
