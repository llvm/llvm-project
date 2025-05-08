// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding < %s 2>&1 | FileCheck --check-prefix=GFX13-ERR --implicit-check-not=error: %s

buffer_load_b32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b32 v5, off, s[8:11], s3 offset:8388607 lds
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b96 v[5:7], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b96 v[5:7], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b96 v[5:7], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b96 v[5:7], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b96 v[5:7], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b96 v[5:7], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b96 v[5:7], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b96 v[5:7], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b128 v[5:8], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b128 v[5:8], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b128 v[5:8], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b128 v[5:8], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b128 v[5:8], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b128 v[5:8], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b128 v[5:8], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_b128 v[5:8], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_b16 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_b16 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_b16 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_b16 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_b16 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_b16 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_b16 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_b16 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_b16 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_b16 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_b16 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_b16 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_b16 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_b16 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_b16 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_b16 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_i8 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_i8 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_i8 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_i8 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_i8 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_i8 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_i8 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_i8 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_u8 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_u8 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_u8 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_u8 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_hi_u8 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_u8 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_u8 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_hi_u8 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_i8 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_i8 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_i8 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_i8 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_i8 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_i8 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_i8 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_i8 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_u8 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_u8 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_u8 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_u8 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_d16_u8 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_u8 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_u8 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_d16_u8 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i8 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_i8 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_i8 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_i8 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_i8 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i8 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i8 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i8 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i16 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_i16 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_i16 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_i16 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_i16 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i16 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i16 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_i16 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u8 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_u8 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_u8 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_u8 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_u8 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u8 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u8 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u8 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u16 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_u16 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_u16 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_u16 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_u16 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u16 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u16 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_load_u16 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b8 v1, off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b8 v1, off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b8 v1, off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b8 v1, off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b8 v1, off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b8 v1, off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b8 v1, off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b8 v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b16 v1, off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b16 v1, off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b16 v1, off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b16 v1, off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b16 v1, off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b16 v1, off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b16 v1, off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b16 v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b32 v1, off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b32 v1, off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b32 v1, off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b32 v1, off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b32 v1, off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b32 v1, off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b32 v1, off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b32 v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b64 v[1:2], off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b64 v[1:2], off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b64 v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b64 v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b64 v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b64 v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b64 v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b64 v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b96 v[1:3], off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b96 v[1:3], off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b96 v[1:3], off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b96 v[1:3], off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b96 v[1:3], off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b96 v[1:3], off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b96 v[1:3], off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b96 v[1:3], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b128 v[1:4], off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b128 v[1:4], off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b128 v[1:4], off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b128 v[1:4], off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b128 v[1:4], off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b128 v[1:4], off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b128 v[1:4], off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_b128 v[1:4], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b8 v1, off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_b8 v1, off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_b8 v1, off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_b8 v1, off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_b8 v1, off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b8 v1, off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b8 v1, off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b8 v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b16 v1, off, s[12:15], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_b16 v1, off, s[12:15], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_b16 v1, off, s[12:15], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_b16 v1, off, s[12:15], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_d16_hi_b16 v1, off, s[12:15], s4 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b16 v1, off, s[12:15], s4 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b16 v1, off, s[12:15], s4 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_store_d16_hi_b16 v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_f16 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_pk_add_f16 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_pk_add_f16 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_pk_add_f16 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_pk_add_f16 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_f16 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_f16 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_f16 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_bf16 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_pk_add_bf16 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_pk_add_bf16 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_pk_add_bf16 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_f32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_f32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_f32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_f32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_f32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_f32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_f32 v5, off, s[8:11], s3 offset:8388607 glc slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_u32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_u32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_u32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_u32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_u64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_u64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_u64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_add_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_add_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_and_b32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_and_b32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_and_b32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_and_b32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_and_b64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_and_b64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_and_b64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_and_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_and_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b32 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_cmpswap_b32 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_cmpswap_b32 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_cmpswap_b32 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_cmpswap_b32 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b32 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b32 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b32 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b64 v[5:8], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_cmpswap_b64 v[5:8], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_cmpswap_b64 v[5:8], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_cmpswap_b64 v[5:8], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_cmpswap_b64 v[5:8], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b64 v[5:8], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b64 v[5:8], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cmpswap_b64 v[5:8], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v255, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[12:15], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[96:99], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], s101 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], m0 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], 0 offset:8388607 th:TH_ATOMIC_RETURN
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], 0 offset:8388607 th:TH_ATOMIC_RT_RETURN scope:SCOPE_SE
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], 0 offset:8388607 th:TH_ATOMIC_CASCADE_NT scope:SCOPE_DEV
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], 0 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], -1 offset:8388607 th:TH_ATOMIC_RETURN
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], -1 offset:8388607 th:TH_ATOMIC_RT_RETURN scope:SCOPE_SE
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], -1 offset:8388607 th:TH_ATOMIC_CASCADE_NT scope:SCOPE_DEV
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], -1 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], 0.5 offset:8388607 th:TH_ATOMIC_RETURN
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], 0.5 offset:8388607 th:TH_ATOMIC_RT_RETURN scope:SCOPE_SE
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], 0.5 offset:8388607 th:TH_ATOMIC_CASCADE_NT scope:SCOPE_DEV
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], 0.5 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], -4.0 offset:8388607 th:TH_ATOMIC_RETURN
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], -4.0 offset:8388607 th:TH_ATOMIC_RT_RETURN scope:SCOPE_SE
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], -4.0 offset:8388607 th:TH_ATOMIC_CASCADE_NT scope:SCOPE_DEV
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], -4.0 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, v0, s[8:11], s3 idxen offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, v0, s[8:11], s3 offen offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], s3 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], s3 offset:0 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], s3 offset:7 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], s3 offset:8388607 glc slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], s3 offset:8388607 glc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_clamp_u32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cond_sub_u32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cond_sub_u32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cond_sub_u32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cond_sub_u32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cond_sub_u32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cond_sub_u32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cond_sub_u32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_cond_sub_u32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_dec_u32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_dec_u32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_dec_u32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_dec_u32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_dec_u64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_dec_u64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_dec_u64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_dec_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_dec_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_inc_u32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_inc_u32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_inc_u32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_inc_u32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_inc_u64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_inc_u64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_inc_u64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_inc_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_inc_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_num_f32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_num_f32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_num_f32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_num_f32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_num_f32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_num_f32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_num_f32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_num_f32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_i32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_i32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_i32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_i32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_i64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_i64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_i64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_i64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_i64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_u32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_u32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_u32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_u32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_u64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_u64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_u64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_max_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_max_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_f32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_f32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_f32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_f32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_f32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_f32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_f32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_f32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_i32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_i32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_i32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_i32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_i64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_i64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_i64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_i64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_i64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_u32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_u32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_u32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_u32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_u64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_u64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_u64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_min_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_min_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_or_b32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_or_b32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_or_b32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_or_b32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_or_b64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_or_b64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_or_b64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_or_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_or_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_sub_u32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_sub_u32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_sub_u32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_sub_u32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_sub_u64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_sub_u64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_sub_u64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_sub_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_sub_u64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_swap_b32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_swap_b32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_swap_b32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_swap_b32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_swap_b64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_swap_b64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_swap_b64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_swap_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_swap_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b32 v5, off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_xor_b32 v5, off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_xor_b32 v5, off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_xor_b32 v5, off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_xor_b32 v5, off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b32 v5, off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b32 v5, off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b32 v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b64 v[5:6], off, s[8:11], 0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_xor_b64 v[5:6], off, s[8:11], -1 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_xor_b64 v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_xor_b64 v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_atomic_xor_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b64 v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b64 v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

buffer_atomic_xor_b64 v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
