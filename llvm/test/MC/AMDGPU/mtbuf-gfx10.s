// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck -check-prefix=GFX10-ERR --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Positive tests for legacy format syntax.
//===----------------------------------------------------------------------===//

// GFX10: tbuffer_load_format_d16_x v0, off, s[0:3], 0 format:[BUF_FMT_32_FLOAT] ; encoding: [0x00,0x00,0xb0,0xe8,0x00,0x00,0x20,0x80]
tbuffer_load_format_d16_x v0, off, s[0:3], format:22, 0

// GFX10: tbuffer_load_format_d16_xy v0, off, s[0:3], 0 format:[BUF_FMT_32_FLOAT] ; encoding: [0x00,0x00,0xb1,0xe8,0x00,0x00,0x20,0x80]
tbuffer_load_format_d16_xy v0, off, s[0:3], format:22, 0

// GFX10: tbuffer_load_format_d16_xyzw v[0:1], off, s[0:3], 0 format:[BUF_FMT_32_FLOAT] ; encoding: [0x00,0x00,0xb3,0xe8,0x00,0x00,0x20,0x80]
tbuffer_load_format_d16_xyzw v[0:1], off, s[0:3], format:22, 0

// GFX10: tbuffer_load_format_xyzw v[0:3], off, s[0:3], 0 format:78 ; encoding: [0x00,0x00,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], off, s[0:3], format:78, 0

// GFX10: tbuffer_load_format_xyzw v[8:11], off, s[0:3], 0 format:[BUF_FMT_32_FLOAT] slc ; encoding: [0x00,0x00,0xb3,0xe8,0x00,0x08,0x40,0x80]
tbuffer_load_format_xyzw v[8:11], off, s[0:3], format:22, 0 slc

// GFX10: tbuffer_load_format_xyzw v[4:7], off, s[0:3], 0 format:[BUF_FMT_32_32_SINT] glc ; encoding: [0x00,0x40,0xfb,0xe9,0x00,0x04,0x00,0x80]
tbuffer_load_format_xyzw v[4:7], off, s[0:3], format:63, 0 glc

// GFX10: tbuffer_load_format_xyzw v[12:15], off, s[0:3], 0 format:[BUF_FMT_16_16_UNORM] glc dlc ; encoding: [0x00,0xc0,0xbb,0xe8,0x00,0x0c,0x00,0x80]
tbuffer_load_format_xyzw v[12:15], off, s[0:3], format:23, 0 glc dlc

// GFX10: tbuffer_load_format_xyzw v[0:3], off, s[0:3], 0 format:78 offset:42 ; encoding: [0x2a,0x00,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], off, s[0:3], format:78, 0 offset:42

// GFX10: tbuffer_load_format_xyzw v[4:7], off, s[0:3], s4 format:[BUF_FMT_32_32_UINT] offset:73 ; encoding: [0x49,0x00,0xf3,0xe9,0x00,0x04,0x00,0x04]
tbuffer_load_format_xyzw v[4:7], off, s[0:3], format:62, s4 offset:73

// GFX10: tbuffer_load_format_xyzw v[0:3], off, s[0:3], 61 format:[BUF_FMT_10_10_10_2_SSCALED] offset:4095 ; encoding: [0xff,0x0f,0x7b,0xe9,0x00,0x00,0x00,0xbd]
tbuffer_load_format_xyzw v[0:3], off, s[0:3], format:47, 61 offset:4095

// GFX10: tbuffer_load_format_xyzw v[8:11], off, s[0:3], s4 format:[BUF_FMT_32_32_32_32_FLOAT] offset:1 ; encoding: [0x01,0x00,0x6b,0xea,0x00,0x08,0x00,0x04]
tbuffer_load_format_xyzw v[8:11], off, s[0:3], format:77, s4 offset:1

// GFX10: tbuffer_load_format_xyzw v[0:3], v0, s[0:3], 0 format:78 idxen ; encoding: [0x00,0x20,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], v0, s[0:3], format:78, 0 idxen

// GFX10: tbuffer_load_format_xyzw v[0:3], v0, s[0:3], 0 format:78 offen ; encoding: [0x00,0x10,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], v0, s[0:3], format:78, 0 offen

// GFX10: tbuffer_load_format_xyzw v[0:3], v0, s[0:3], 0 format:78 offen offset:52 ; encoding: [0x34,0x10,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], v0, s[0:3], format:78, 0 offen offset:52

// GFX10: tbuffer_load_format_xyzw v[0:3], v[0:1], s[0:3], 0 format:78 idxen offen ; encoding: [0x00,0x30,0x73,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xyzw v[0:3], v[0:1], s[0:3], format:78, 0 idxen offen

// GFX10: tbuffer_load_format_xy v[0:1], off, s[0:3], 0 format:[BUF_FMT_32_32_32_32_FLOAT] ; encoding: [0x00,0x00,0x69,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_xy v[0:1], off, s[0:3], format:77, 0

// GFX10: tbuffer_load_format_x v0, off, s[0:3], 0 format:[BUF_FMT_32_32_32_32_FLOAT] ; encoding: [0x00,0x00,0x68,0xea,0x00,0x00,0x00,0x80]
tbuffer_load_format_x v0, off, s[0:3], format:77, 0

// GFX10: tbuffer_store_format_d16_x v0, v1, s[4:7], 0 format:[BUF_FMT_10_11_11_SSCALED] idxen ; encoding: [0x00,0x20,0x0c,0xe9,0x01,0x00,0x21,0x80]
tbuffer_store_format_d16_x v0, v1, s[4:7], format:33, 0 idxen

// GFX10: tbuffer_store_format_d16_xy v0, v1, s[4:7], 0 format:[BUF_FMT_10_11_11_SSCALED] idxen ; encoding: [0x00,0x20,0x0d,0xe9,0x01,0x00,0x21,0x80]
tbuffer_store_format_d16_xy v0, v1, s[4:7], format:33, 0 idxen

// GFX10: tbuffer_store_format_d16_xyzw v[0:1], v2, s[4:7], 0 format:[BUF_FMT_10_11_11_SSCALED] idxen ; encoding: [0x00,0x20,0x0f,0xe9,0x02,0x00,0x21,0x80]
tbuffer_store_format_d16_xyzw v[0:1], v2, s[4:7], format:33, 0 idxen

// GFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], 0 format:[BUF_FMT_10_10_10_2_UNORM] ; encoding: [0x00,0x00,0x67,0xe9,0x00,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], off, s[0:3], format:44, 0

// GFX10: tbuffer_store_format_xyzw v[4:7], off, s[0:3], 0 format:[BUF_FMT_8_8_8_8_SINT] glc ; encoding: [0x00,0x40,0xef,0xe9,0x00,0x04,0x00,0x80]
tbuffer_store_format_xyzw v[4:7], off, s[0:3], format:61, 0 glc

// GFX10: tbuffer_store_format_xyzw v[8:11], off, s[0:3], 0 format:78 slc ; encoding: [0x00,0x00,0x77,0xea,0x00,0x08,0x40,0x80]
tbuffer_store_format_xyzw v[8:11], off, s[0:3], format:78, 0 slc

// GFX10: tbuffer_store_format_xyzw v[8:11], off, s[0:3], 0 format:78 ; encoding: [0x00,0x00,0x77,0xea,0x00,0x08,0x00,0x80]
tbuffer_store_format_xyzw v[8:11], off, s[0:3], format:78, 0

// GFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], 0 format:117 offset:42 ; encoding: [0x2a,0x00,0xaf,0xeb,0x00,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], off, s[0:3], format:117, 0 offset:42

// GFX10: tbuffer_store_format_xyzw v[0:3], off, s[0:3], s4 format:117 offset:42 ; encoding: [0x2a,0x00,0xaf,0xeb,0x00,0x00,0x00,0x04]
tbuffer_store_format_xyzw v[0:3], off, s[0:3], format:117, s4 offset:42

// GFX10: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], 0 format:[BUF_FMT_10_10_10_2_SSCALED] idxen ; encoding: [0x00,0x20,0x7f,0xe9,0x04,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v4, s[0:3], format:47, 0 idxen

// GFX10: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], 0 format:115 offen ; encoding: [0x00,0x10,0x9f,0xeb,0x04,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v4, s[0:3], format:115, 0 offen

// GFX10: tbuffer_store_format_xyzw v[0:3], v[4:5], s[0:3], 0 format:[BUF_FMT_16_16_16_16_SINT] idxen offen ; encoding: [0x00,0x30,0x37,0xea,0x04,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v[4:5], s[0:3], format:70, 0 idxen offen

// GFX10: tbuffer_store_format_xyzw v[0:3], v4, s[0:3], 0 format:[BUF_FMT_32_32_SINT] idxen ; encoding: [0x00,0x20,0xff,0xe9,0x04,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v4, s[0:3], format:63, 0 idxen

// GFX10: tbuffer_store_format_xyzw v[0:3], v6, s[0:3], 0 format:[BUF_FMT_10_10_10_2_USCALED] idxen ; encoding: [0x00,0x20,0x77,0xe9,0x06,0x00,0x00,0x80]
tbuffer_store_format_xyzw v[0:3], v6, s[0:3], format:46, 0 idxen

// GFX10: tbuffer_store_format_x v0, v1, s[0:3], 0 format:125 idxen ; encoding: [0x00,0x20,0xec,0xeb,0x01,0x00,0x00,0x80]
tbuffer_store_format_x v0, v1, s[0:3], format:125, 0 idxen

// GFX10: tbuffer_store_format_xy v[0:1], v2, s[0:3], 0 format:[BUF_FMT_10_11_11_SSCALED] idxen ; encoding: [0x00,0x20,0x0d,0xe9,0x02,0x00,0x00,0x80]
tbuffer_store_format_xy v[0:1], v2, s[0:3], format:33, 0 idxen

// GFX10: tbuffer_store_format_x v0, v1, s[0:3], 0 format:127 idxen ; encoding: [0x00,0x20,0xfc,0xeb,0x01,0x00,0x00,0x80]
tbuffer_store_format_x v0, v1, s[0:3], format:127, 0 idxen

// GFX10: tbuffer_store_format_x v0, v1, s[0:3], 0 format:127 idxen ; encoding: [0x00,0x20,0xfc,0xeb,0x01,0x00,0x00,0x80]
tbuffer_store_format_x v0, v1, s[0:3] format:127 0 idxen

// GFX10: tbuffer_store_format_x v0, v1, s[0:3], s0 format:[BUF_FMT_INVALID] idxen ; encoding: [0x00,0x20,0x04,0xe8,0x01,0x00,0x00,0x00]
tbuffer_store_format_x v0, v1, s[0:3] format:0 s0 idxen

// GFX10: tbuffer_store_format_x v0, v1, s[0:3], s0 idxen ; encoding: [0x00,0x20,0x0c,0xe8,0x01,0x00,0x00,0x00]
tbuffer_store_format_x v0, v1, s[0:3] format:1 s0 idxen

// GFX10: tbuffer_store_format_x v0, v1, s[0:3], 0 idxen ; encoding: [0x00,0x20,0x0c,0xe8,0x01,0x00,0x00,0x80]
tbuffer_store_format_x v0, v1, s[0:3], 0 idxen

// GFX10: tbuffer_load_format_d16_x v0, off, s[0:3], s0 ; encoding: [0x00,0x00,0x08,0xe8,0x00,0x00,0x20,0x00]
tbuffer_load_format_d16_x v0, off, s[0:3] s0

MAX_FORMAT=127
// GFX10: tbuffer_store_format_x v0, v1, s[0:3], s0 format:[BUF_FMT_INVALID] idxen ; encoding: [0x00,0x20,0x04,0xe8,0x01,0x00,0x00,0x00]
tbuffer_store_format_x v0, v1, s[0:3] format:0 s0 idxen

// GFX10: tbuffer_store_format_x v0, v1, s[0:3], s0 format:[BUF_FMT_INVALID] idxen ; encoding: [0x00,0x20,0x04,0xe8,0x01,0x00,0x00,0x00]
tbuffer_store_format_x v0, v1, s[0:3] format:0 s0 idxen

//===----------------------------------------------------------------------===//
// Negative tests for legacy format syntax.
//===----------------------------------------------------------------------===//

// GFX10-ERR: error: out of range format
tbuffer_load_format_d16_x v0, off, s[0:3], format:-1, 0

// GFX10-ERR: error: out of range format
tbuffer_load_format_d16_x v0, off, s[0:3], format:128, s0

// GFX10-ERR: error: too few operands for instruction
tbuffer_load_format_d16_x v0, off, s[0:3], format:127

// GFX10-ERR: error: too few operands for instruction
tbuffer_load_format_d16_x v0, off, s[0:3]

// GFX10-ERR: error: invalid operand for instruction
tbuffer_load_format_d16_x v0, off, s[0:3] idxen

// GFX10-ERR: error: unknown token in expression
tbuffer_load_format_d16_x v0, off, s[0:3], format:1,, s0

// GFX10-ERR: error: unknown token in expression
tbuffer_load_format_d16_x v0, off, s[0:3], format:1:, s0

// GFX10-ERR: error: unknown token in expression
tbuffer_load_format_d16_x v0, off, s[0:3],, format:1, s0

//===----------------------------------------------------------------------===//
// Positive tests for symbolic MTBUF format.
//===----------------------------------------------------------------------===//

// Format may be specified in numeric form (min value).
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:0 idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_INVALID] idxen ; encoding: [0x00,0x20,0x07,0xe8,0x01,0x01,0x01,0x00]

// Format may be specified in numeric form (max value).
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:127 idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:127 idxen ; encoding: [0x00,0x20,0xff,0xeb,0x01,0x01,0x01,0x00]

// Format may be specified in numeric form (first unsupported value).
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:78 idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:78 idxen ; encoding: [0x00,0x20,0x77,0xea,0x01,0x01,0x01,0x00]

// Format may be specified as an expression.
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:(2 + 3 * 16) idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_UNORM] idxen ; encoding: [0x00,0x20,0x97,0xe9,0x01,0x01,0x01,0x00]

// format may be specified as a list of dfmt, nfmt:
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_8,BUF_NUM_FORMAT_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 idxen ; encoding: [0x00,0x20,0x0f,0xe8,0x01,0x01,0x01,0x00]

// nfmt and dfmt can be in either order:
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_NUM_FORMAT_SNORM, BUF_DATA_FORMAT_16] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_SNORM] idxen ; encoding: [0x00,0x20,0x47,0xe8,0x01,0x01,0x01,0x00]

// nfmt may be omitted:
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[ BUF_DATA_FORMAT_8_8 ] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_UNORM] idxen ; encoding: [0x00,0x20,0x77,0xe8,0x01,0x01,0x01,0x00]

// dfmt may be omitted:
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_NUM_FORMAT_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_USCALED] idxen ; encoding: [0x00,0x20,0x1f,0xe8,0x01,0x01,0x01,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_16_16] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_UNORM] idxen ; encoding: [0x00,0x20,0xbf,0xe8,0x01,0x01,0x01,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_10_11_11] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_UNORM] idxen ; encoding: [0x00,0x20,0xf7,0xe8,0x01,0x01,0x01,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_11_11_10] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_UNORM] idxen ; encoding: [0x00,0x20,0x2f,0xe9,0x01,0x01,0x01,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_10_10_10_2] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_UNORM] idxen ; encoding: [0x00,0x20,0x67,0xe9,0x01,0x01,0x01,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_2_10_10_10] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_UNORM] idxen ; encoding: [0x00,0x20,0x97,0xe9,0x01,0x01,0x01,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_8_8_8_8] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_UNORM] idxen ; encoding: [0x00,0x20,0xc7,0xe9,0x01,0x01,0x01,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_16_16_16_16] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_UNORM] idxen ; encoding: [0x00,0x20,0x0f,0xea,0x01,0x01,0x01,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_INVALID] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_INVALID] idxen ; encoding: [0x00,0x20,0x07,0xe8,0x01,0x01,0x01,0x00]

// Check nfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_NUM_FORMAT_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SSCALED] idxen ; encoding: [0x00,0x20,0x27,0xe8,0x01,0x01,0x01,0x00]

// Check nfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_NUM_FORMAT_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_UINT] idxen ; encoding: [0x00,0x20,0x2f,0xe8,0x01,0x01,0x01,0x00]

// Check nfmt formats
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_NUM_FORMAT_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SINT] idxen ; encoding: [0x00,0x20,0x37,0xe8,0x01,0x01,0x01,0x00]

//===----------------------------------------------------------------------===//
// Negative tests for symbolic format errors handling.
//===----------------------------------------------------------------------===//

// Unknown format specifier
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT] idxen
// GFX10-ERR: error: unsupported format

// Valid but unsupported format specifier (SNORM_OGL is supported for SI/CI only)
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_NUM_FORMAT_SNORM_OGL] idxen
// GFX10-ERR: error: unsupported format

// Valid but unsupported format specifier (RESERVED_6 is supported for VI/GFX9 only)
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_NUM_FORMAT_RESERVED_6] idxen
// GFX10-ERR: error: unsupported format

// Unsupported format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_32] idxen
// GFX10-ERR: error: unsupported format

// Unsupported format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_32_32] idxen
// GFX10-ERR: error: unsupported format

// Unsupported format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_32_32_32] idxen
// GFX10-ERR: error: unsupported format

// Unsupported format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_32_32_32_32] idxen
// GFX10-ERR: error: unsupported format

// Unsupported format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_32_32_32_32, BUF_NUM_FORMAT_UNORM] idxen
// GFX10-ERR: error: unsupported format

// Unsupported format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_RESERVED_15] idxen
// GFX10-ERR: error: unsupported format

// Unsupported format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_NUM_FORMAT_FLOAT] idxen
// GFX10-ERR: error: unsupported format

// Unsupported format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_FLOAT] idxen
// GFX10-ERR: error: unsupported format

//===----------------------------------------------------------------------===//
// Positive tests for unified MTBUF format (GFX10+).
//===----------------------------------------------------------------------===//

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_INVALID] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_INVALID] idxen ; encoding: [0x00,0x20,0x07,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 idxen ; encoding: [0x00,0x20,0x0f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SNORM] idxen ; encoding: [0x00,0x20,0x17,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_USCALED] idxen ; encoding: [0x00,0x20,0x1f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SSCALED] idxen ; encoding: [0x00,0x20,0x27,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_UINT] idxen ; encoding: [0x00,0x20,0x2f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SINT] idxen ; encoding: [0x00,0x20,0x37,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_UNORM] idxen ; encoding: [0x00,0x20,0x3f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_SNORM] idxen ; encoding: [0x00,0x20,0x47,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_USCALED] idxen ; encoding: [0x00,0x20,0x4f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_SSCALED] idxen ; encoding: [0x00,0x20,0x57,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_UINT] idxen ; encoding: [0x00,0x20,0x5f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_SINT] idxen ; encoding: [0x00,0x20,0x67,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_FLOAT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_FLOAT] idxen ; encoding: [0x00,0x20,0x6f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_UNORM] idxen ; encoding: [0x00,0x20,0x77,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_SNORM] idxen ; encoding: [0x00,0x20,0x7f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_USCALED] idxen ; encoding: [0x00,0x20,0x87,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_SSCALED] idxen ; encoding: [0x00,0x20,0x8f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_UINT] idxen ; encoding: [0x00,0x20,0x97,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_SINT] idxen ; encoding: [0x00,0x20,0x9f,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_UINT] idxen ; encoding: [0x00,0x20,0xa7,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_SINT] idxen ; encoding: [0x00,0x20,0xaf,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_FLOAT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_FLOAT] idxen ; encoding: [0x00,0x20,0xb7,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_UNORM] idxen ; encoding: [0x00,0x20,0xbf,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_SNORM] idxen ; encoding: [0x00,0x20,0xc7,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_USCALED] idxen ; encoding: [0x00,0x20,0xcf,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_SSCALED] idxen ; encoding: [0x00,0x20,0xd7,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_UINT] idxen ; encoding: [0x00,0x20,0xdf,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_SINT] idxen ; encoding: [0x00,0x20,0xe7,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_FLOAT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_FLOAT] idxen ; encoding: [0x00,0x20,0xef,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_UNORM] idxen ; encoding: [0x00,0x20,0xf7,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_SNORM] idxen ; encoding: [0x00,0x20,0xff,0xe8,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_USCALED] idxen ; encoding: [0x00,0x20,0x07,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_SSCALED] idxen ; encoding: [0x00,0x20,0x0f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_UINT] idxen ; encoding: [0x00,0x20,0x17,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_SINT] idxen ; encoding: [0x00,0x20,0x1f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_FLOAT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_11_11_FLOAT] idxen ; encoding: [0x00,0x20,0x27,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_UNORM] idxen ; encoding: [0x00,0x20,0x2f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_SNORM] idxen ; encoding: [0x00,0x20,0x37,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_USCALED] idxen ; encoding: [0x00,0x20,0x3f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_SSCALED] idxen ; encoding: [0x00,0x20,0x47,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_UINT] idxen ; encoding: [0x00,0x20,0x4f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_SINT] idxen ; encoding: [0x00,0x20,0x57,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_FLOAT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_11_11_10_FLOAT] idxen ; encoding: [0x00,0x20,0x5f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_UNORM] idxen ; encoding: [0x00,0x20,0x67,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_SNORM] idxen ; encoding: [0x00,0x20,0x6f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_USCALED] idxen ; encoding: [0x00,0x20,0x77,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_SSCALED] idxen ; encoding: [0x00,0x20,0x7f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_UINT] idxen ; encoding: [0x00,0x20,0x87,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_10_10_10_2_SINT] idxen ; encoding: [0x00,0x20,0x8f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_UNORM] idxen ; encoding: [0x00,0x20,0x97,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_SNORM] idxen ; encoding: [0x00,0x20,0x9f,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_USCALED] idxen ; encoding: [0x00,0x20,0xa7,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_SSCALED] idxen ; encoding: [0x00,0x20,0xaf,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_UINT] idxen ; encoding: [0x00,0x20,0xb7,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_2_10_10_10_SINT] idxen ; encoding: [0x00,0x20,0xbf,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_UNORM] idxen ; encoding: [0x00,0x20,0xc7,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_SNORM] idxen ; encoding: [0x00,0x20,0xcf,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_USCALED] idxen ; encoding: [0x00,0x20,0xd7,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_SSCALED] idxen ; encoding: [0x00,0x20,0xdf,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_UINT] idxen ; encoding: [0x00,0x20,0xe7,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_8_8_8_SINT] idxen ; encoding: [0x00,0x20,0xef,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_UINT] idxen ; encoding: [0x00,0x20,0xf7,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_SINT] idxen ; encoding: [0x00,0x20,0xff,0xe9,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_FLOAT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_FLOAT] idxen ; encoding: [0x00,0x20,0x07,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_UNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_UNORM] idxen ; encoding: [0x00,0x20,0x0f,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_SNORM] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_SNORM] idxen ; encoding: [0x00,0x20,0x17,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_USCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_USCALED] idxen ; encoding: [0x00,0x20,0x1f,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_SSCALED] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_SSCALED] idxen ; encoding: [0x00,0x20,0x27,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_UINT] idxen ; encoding: [0x00,0x20,0x2f,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_SINT] idxen ; encoding: [0x00,0x20,0x37,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_FLOAT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_16_16_16_16_FLOAT] idxen ; encoding: [0x00,0x20,0x3f,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_UINT] idxen ; encoding: [0x00,0x20,0x47,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_SINT] idxen ; encoding: [0x00,0x20,0x4f,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_FLOAT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_FLOAT] idxen ; encoding: [0x00,0x20,0x57,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_32_UINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_32_UINT] idxen ; encoding: [0x00,0x20,0x5f,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_32_SINT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_32_SINT] idxen ; encoding: [0x00,0x20,0x67,0xea,0x01,0x01,0x01,0x00]

tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_32_FLOAT] idxen
// GFX10: tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_32_32_32_32_FLOAT] idxen ; encoding: [0x00,0x20,0x6f,0xea,0x01,0x01,0x01,0x00]

//===----------------------------------------------------------------------===//
// Negative tests for unified MTBUF format (GFX10+).
//===----------------------------------------------------------------------===//

// Excessive commas
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SNORM,] idxen
// GFX10-ERR: error: expected a closing square bracket

// Duplicate format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SNORM,BUF_FMT_8_SNORM] idxen
// GFX10-ERR: error: expected a closing square bracket

// Duplicate format
tbuffer_store_format_xyzw v[1:4], v1, s[4:7], s0 format:[BUF_FMT_8_SNORM,BUF_DATA_FORMAT_8] idxen
// GFX10-ERR: error: expected a closing square bracket
