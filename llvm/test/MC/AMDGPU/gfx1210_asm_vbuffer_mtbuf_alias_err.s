// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

tbuffer_load_format_d16_x v4, off, s[8:11], s3 format:[BUF_FMT_8_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_d16_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_d16_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_d16_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_8_8_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_d16_x v4, off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_d16_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_8_8_8_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_d16_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_d16_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
