// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding < %s 2>&1 | FileCheck --check-prefix=GFX13-ERR --implicit-check-not=error: %s

buffer_load_d16_format_x v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_x v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_x v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_x v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xy v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xy v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xy v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xy v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyz v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyz v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyz v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyz v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_format_x v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_format_x v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_format_x v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_format_x v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_x v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_x v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_x v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xy v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xy v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xy v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xy v[5:7], off, s[8:11], s3 offset:8388607 glc slc dlc tfe
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[5:7], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyz v[5:7], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyz v[5:7], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyz v[5:7], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[5:8], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyzw v[5:8], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyzw v[5:8], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyzw v[5:8], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_x v1, off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_x v1, off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_x v1, off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_x v1, off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xy v1, off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xy v1, off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xy v1, off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xy v1, off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyz v[1:2], off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyz v[1:2], off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyz v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyz v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_format_x v1, off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_format_x v1, off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_format_x v1, off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_format_x v1, off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_x v1, off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_x v1, off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_x v1, off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xy v[1:2], off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xy v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xy v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyz v[1:3], off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyz v[1:3], off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyz v[1:3], off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyzw v[1:4], off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyzw v[1:4], off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyzw v[1:4], off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

