// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding < %s 2>&1 | FileCheck --check-prefix=GFX13-ERR --implicit-check-not=error: %s

tbuffer_load_d16_format_x v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_x v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_USCALED] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_USCALED] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_USCALED] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_USCALED] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_10_11_11, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_11_11, BUF_NUM_FORMAT_FLOAT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_11_11, BUF_NUM_FORMAT_FLOAT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_11_11, BUF_NUM_FORMAT_FLOAT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SINT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SINT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SINT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_x v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_x v4, off, ttmp[4:7], 0 format:[BUF_FMT_8_SINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_SINT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_SINT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_SINT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], 0 format:[BUF_FMT_32_SINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_SINT] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_SINT] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_SINT] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], 0 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], 0 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

// Removed formats (compared to gfx10)

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_UNORM] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_SNORM] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_USCALED] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_SSCALED] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_UINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_SINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_UNORM] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_SNORM] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_USCALED] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_SSCALED] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_UINT] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_10_10_2_USCALED] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_10_10_2_SSCALED] offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: unsupported format
