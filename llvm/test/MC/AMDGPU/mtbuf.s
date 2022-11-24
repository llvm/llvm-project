// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck -check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck -check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=VI %s

// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s  2>&1 | FileCheck -check-prefixes=GCN-ERR,SICI-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck -check-prefixes=GCN-ERR,SICI-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s   2>&1 | FileCheck -check-prefixes=GCN-ERR,VI-ERR --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Positive tests for legacy dfmt/nfmt syntax.
//===----------------------------------------------------------------------===//

tbuffer_load_format_x v1, off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_load_format_x v1, off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x78,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_load_format_x v1, off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x78,0xe9,0x00,0x01,0x01,0x01]

tbuffer_load_format_xy v[1:2], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_load_format_xy v[1:2], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x79,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_load_format_xy v[1:2], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x78,0xe9,0x00,0x01,0x01,0x01]

tbuffer_load_format_xyz v[1:3], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_load_format_xyz v[1:3], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x7a,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_load_format_xyz v[1:3], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x79,0xe9,0x00,0x01,0x01,0x01]

tbuffer_load_format_xyzw v[1:4], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_load_format_xyzw v[1:4], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x7b,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_load_format_xyzw v[1:4], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x79,0xe9,0x00,0x01,0x01,0x01]

tbuffer_store_format_x v1, off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_store_format_x v1, off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x7c,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_store_format_x v1, off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x7a,0xe9,0x00,0x01,0x01,0x01]

tbuffer_store_format_xy v[1:2], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_store_format_xy v[1:2], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x7d,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_store_format_xy v[1:2], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x7a,0xe9,0x00,0x01,0x01,0x01]

tbuffer_store_format_xyzw v[1:4], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_store_format_xyzw v[1:4], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x7f,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_store_format_xyzw v[1:4], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x7b,0xe9,0x00,0x01,0x01,0x01]

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, nfmt:2, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x7f,0xe9,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x7b,0xe9,0x00,0x01,0x1d,0x71]

// nfmt is optional:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15] ; encoding: [0x00,0x00,0x7f,0xe8,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15] ; encoding: [0x00,0x80,0x7b,0xe8,0x00,0x01,0x1d,0x71]

// dfmt is optional:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], nfmt:2, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x0f,0xe9,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x0b,0xe9,0x00,0x01,0x1d,0x71]

// nfmt and dfmt can be in either order:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], nfmt:2, dfmt:15, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x7f,0xe9,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x7b,0xe9,0x00,0x01,0x1d,0x71]

// nfmt and dfmt may be omitted:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x0f,0xe8,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x80,0x0b,0xe8,0x00,0x01,0x1d,0x71]

// Check dfmt/nfmt min values
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:0, nfmt:0, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_INVALID] ; encoding: [0x00,0x00,0x07,0xe8,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_INVALID] ; encoding: [0x00,0x80,0x03,0xe8,0x00,0x01,0x1d,0x71]

// Check dfmt/nfmt max values
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, nfmt:7, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x00,0xff,0xeb,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x80,0xfb,0xeb,0x00,0x01,0x1d,0x71]

// Check default dfmt/nfmt values
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:1, nfmt:0, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x0f,0xe8,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x80,0x0b,0xe8,0x00,0x01,0x1d,0x71]

// Check that comma separators are optional
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:15 nfmt:7 ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x00,0xff,0xeb,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x80,0xfb,0xeb,0x00,0x01,0x1d,0x71]

dfmt=15
nfmt=7
// Check expressions with dfmt
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:-1+dfmt+1 nfmt:nfmt ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x00,0xff,0xeb,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x80,0xfb,0xeb,0x00,0x01,0x1d,0x71]

//===----------------------------------------------------------------------===//
// Negative tests for legacy dfmt/nfmt syntax.
//===----------------------------------------------------------------------===//

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:-1 nfmt:1 s0
// GCN-ERR: error: out of range dfmt

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:16 nfmt:1 s0
// GCN-ERR: error: out of range dfmt

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:-1 s0
// GCN-ERR: error: out of range nfmt

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:8 s0
// GCN-ERR: error: out of range nfmt

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7]
// GCN-ERR: error: too few operands for instruction

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7],, dfmt:1 nfmt:1 s0
// GCN-ERR: error: unknown token in expression

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1,, nfmt:1 s0
// GCN-ERR: error: unknown token in expression

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:1,, s0
// GCN-ERR: error: unknown token in expression

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 dfmt:1 s0
// GCN-ERR: error: invalid operand for instruction

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] nfmt:1 nfmt:1 s0
// GCN-ERR: error: invalid operand for instruction

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:1 dfmt:1 s0
// GCN-ERR: error: invalid operand for instruction

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] nfmt:1 dfmt:1 nfmt:1 s0
// GCN-ERR: error: invalid operand for instruction

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1: nfmt:1 s0
// GCN-ERR: error: unknown token in expression

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:1: s0
// GCN-ERR: error: unknown token in expression

//===----------------------------------------------------------------------===//
// Tests for symbolic MTBUF format
//===----------------------------------------------------------------------===//

// Format may be specified in numeric form (min value).
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:0
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_INVALID] ; encoding: [0x00,0x00,0x07,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_INVALID] ; encoding: [0x00,0x80,0x03,0xe8,0x00,0x01,0x1d,0x00]

// Format may be specified in numeric form (max value).
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:127
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x00,0xff,0xeb,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x80,0xfb,0xeb,0x00,0x01,0x1d,0x00]

// Format may be specified as an expression.
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:(2 + 3 * 16)
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16,BUF_NUM_FORMAT_SSCALED] ; encoding: [0x00,0x00,0x97,0xe9,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16,BUF_NUM_FORMAT_SSCALED] ; encoding: [0x00,0x80,0x93,0xe9,0x00,0x01,0x1d,0x00]

// format may be specified as a list of dfmt, nfmt:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_8,BUF_NUM_FORMAT_UNORM]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 ; encoding: [0x00,0x00,0x0f,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 ; encoding: [0x00,0x80,0x0b,0xe8,0x00,0x01,0x1d,0x00]

// nfmt and dfmt can be in either order:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_SNORM, BUF_DATA_FORMAT_16]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16,BUF_NUM_FORMAT_SNORM] ; encoding: [0x00,0x00,0x97,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16,BUF_NUM_FORMAT_SNORM] ; encoding: [0x00,0x80,0x93,0xe8,0x00,0x01,0x1d,0x00]

// nfmt may be omitted:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[ BUF_DATA_FORMAT_8_8 ]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_8_8] ; encoding: [0x00,0x00,0x1f,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_8_8] ; encoding: [0x00,0x80,0x1b,0xe8,0x00,0x01,0x1d,0x00]

// dfmt may be omitted:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_USCALED]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x0f,0xe9,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x0b,0xe9,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32] ; encoding: [0x00,0x00,0x27,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32] ; encoding: [0x00,0x80,0x23,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16_16]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16_16] ; encoding: [0x00,0x00,0x2f,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16_16] ; encoding: [0x00,0x80,0x2b,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_10_11_11]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_10_11_11] ; encoding: [0x00,0x00,0x37,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_10_11_11] ; encoding: [0x00,0x80,0x33,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_11_11_10]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_11_11_10] ; encoding: [0x00,0x00,0x3f,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_11_11_10] ; encoding: [0x00,0x80,0x3b,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_10_10_10_2]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_10_10_10_2] ; encoding: [0x00,0x00,0x47,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_10_10_10_2] ; encoding: [0x00,0x80,0x43,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_2_10_10_10]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_2_10_10_10] ; encoding: [0x00,0x00,0x4f,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_2_10_10_10] ; encoding: [0x00,0x80,0x4b,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_8_8_8_8]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_8_8_8_8] ; encoding: [0x00,0x00,0x57,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_8_8_8_8] ; encoding: [0x00,0x80,0x53,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32_32]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32_32] ; encoding: [0x00,0x00,0x5f,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32_32] ; encoding: [0x00,0x80,0x5b,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16_16_16_16]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16_16_16_16] ; encoding: [0x00,0x00,0x67,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_16_16_16_16] ; encoding: [0x00,0x80,0x63,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32_32_32]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32_32_32] ; encoding: [0x00,0x00,0x6f,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32_32_32] ; encoding: [0x00,0x80,0x6b,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32_32_32_32]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32_32_32_32] ; encoding: [0x00,0x00,0x77,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32_32_32_32] ; encoding: [0x00,0x80,0x73,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_RESERVED_15]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_RESERVED_15] ; encoding: [0x00,0x00,0x7f,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_RESERVED_15] ; encoding: [0x00,0x80,0x7b,0xe8,0x00,0x01,0x1d,0x00]

// Check dfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_INVALID]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_INVALID] ; encoding: [0x00,0x00,0x07,0xe8,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_INVALID] ; encoding: [0x00,0x80,0x03,0xe8,0x00,0x01,0x1d,0x00]

// Check nfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_SSCALED]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_SSCALED] ; encoding: [0x00,0x00,0x8f,0xe9,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_SSCALED] ; encoding: [0x00,0x80,0x8b,0xe9,0x00,0x01,0x1d,0x00]

// Check nfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT] ; encoding: [0x00,0x00,0x0f,0xea,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT] ; encoding: [0x00,0x80,0x0b,0xea,0x00,0x01,0x1d,0x00]

// Check nfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_SINT]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_SINT] ; encoding: [0x00,0x00,0x8f,0xea,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_SINT] ; encoding: [0x00,0x80,0x8b,0xea,0x00,0x01,0x1d,0x00]

// Check nfmt formats
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_FLOAT]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x00,0x8f,0xeb,0x00,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_FLOAT] ; encoding: [0x00,0x80,0x8b,0xeb,0x00,0x01,0x1d,0x00]

// Check optional comma separators
tbuffer_store_format_xyzw v[1:4], v1, ttmp[4:7], s0, format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT], idxen
// SICI: tbuffer_store_format_xyzw v[1:4], v1, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] idxen ; encoding: [0x00,0x20,0xa7,0xeb,0x01,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], v1, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] idxen ; encoding: [0x00,0xa0,0xa3,0xeb,0x01,0x01,0x1d,0x00]

// Check offen and offset
tbuffer_store_format_xyzw v[1:4], v1, ttmp[4:7], s0, format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] offen offset:52
// SICI: tbuffer_store_format_xyzw v[1:4], v1, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] offen offset:52 ; encoding: [0x34,0x10,0xa7,0xeb,0x01,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], v1, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] offen offset:52 ; encoding: [0x34,0x90,0xa3,0xeb,0x01,0x01,0x1d,0x00]

// Check idxen and offen
tbuffer_store_format_xyzw v[1:4], v[1:2], ttmp[4:7], s0, format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] idxen offen offset:52
// SICI: tbuffer_store_format_xyzw v[1:4], v[1:2], ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] idxen offen offset:52 ; encoding: [0x34,0x30,0xa7,0xeb,0x01,0x01,0x1d,0x00]
// VI: tbuffer_store_format_xyzw v[1:4], v[1:2], ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] idxen offen offset:52 ; encoding: [0x34,0xb0,0xa3,0xeb,0x01,0x01,0x1d,0x00]

// Check addr64
tbuffer_store_format_xyzw v[1:4], v[1:2], ttmp[4:7], s0, format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] addr64
// SICI: tbuffer_store_format_xyzw v[1:4], v[1:2], ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] addr64 ; encoding: [0x00,0x80,0xa7,0xeb,0x01,0x01,0x1d,0x00]
// VI-ERR: error: invalid operand for instruction

//===----------------------------------------------------------------------===//
// Tests for symbolic format errors handling
//===----------------------------------------------------------------------===//

// Missing soffset
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], format:[BUF_DATA_FORMAT_32]
// GCN-ERR: error: not a valid operand.

// Invalid soffset
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s[255] format:[BUF_NUM_FORMAT_FLOAT]
// GCN-ERR: error: register index is out of range

// Both legacy and symbolic formats are specified
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:1 s0 format:[BUF_NUM_FORMAT_FLOAT]
// GCN-ERR: error: duplicate format

// Missing format number
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format: offset:52
// GCN-ERR: error: expected absolute expression

// Invalid number
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:-1
// GCN-ERR: error: out of range format

// Invalid number
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:128
// GCN-ERR: error: out of range format

MAXVAL=127
// Invalid expression
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:MAXVAL+1
// GCN-ERR: error: out of range format

// Empty list
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[]
// GCN-ERR: error: expected a format string

// More than 2 format specifiers
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT,BUF_DATA_FORMAT_8]
// GCN-ERR: error: expected a closing square bracket

// More than 2 format specifiers
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT,BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT]
// GCN-ERR: error: expected a closing square bracket

// Missing brackets
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:BUF_NUM_FORMAT_UINT
// GCN-ERR: error: expected absolute expression

// Unpaired brackets
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT
// GCN-ERR: error: expected a closing square bracket

// Unpaired brackets
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:BUF_NUM_FORMAT_UINT]
// GCN-ERR: error: expected absolute expression

// Missing comma
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT BUF_DATA_FORMAT_32]
// GCN-ERR: error: expected a closing square bracket

// Duplicate dfmt
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_DATA_FORMAT_32]
// GCN-ERR: error: duplicate data format

// Duplicate dfmt
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_32,BUF_DATA_FORMAT_8]
// GCN-ERR: error: duplicate data format

// Duplicate nfmt
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_UINT,BUF_NUM_FORMAT_FLOAT]
// GCN-ERR: error: duplicate numeric format

// Unknown format specifier
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT]
// GCN-ERR: error: unsupported format

// Valid but unsupported format specifier (SNORM_OGL is supported for SI/CI only)
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_SNORM_OGL]
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_SNORM_OGL] ; encoding: [0x00,0x00,0x0f,0xeb,0x00,0x01,0x1d,0x00]
// VI-ERR: error: unsupported format

// Valid but unsupported format specifier (RESERVED_6 is supported for VI/GFX9 only)
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_RESERVED_6]
// SICI-ERR: error: unsupported format
// VI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_NUM_FORMAT_RESERVED_6] ; encoding: [0x00,0x80,0x0b,0xeb,0x00,0x01,0x1d,0x00]

// Excessive commas
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7],, s0 format:[BUF_DATA_FORMAT_8]
// GCN-ERR: error: unknown token in expression

// Excessive commas
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0,, format:[BUF_DATA_FORMAT_8]
// GCN-ERR: error: not a valid operand.

// Excessive commas
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], s0 format:[BUF_DATA_FORMAT_8],, offset:52
// GCN-ERR: error: unknown token in expression
