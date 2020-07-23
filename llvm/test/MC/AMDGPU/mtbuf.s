// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s  2>&1 | FileCheck -check-prefix=GCN-ERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s 2>&1 | FileCheck -check-prefix=GCN-ERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s   2>&1 | FileCheck -check-prefix=GCN-ERR %s

//===----------------------------------------------------------------------===//
// Test for dfmt and nfmt (tbuffer only)
//===----------------------------------------------------------------------===//

tbuffer_load_format_x v1, off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_load_format_x v1, off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x78,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_load_format_x v1, off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x78,0xe9,0x00,0x01,0x01,0x01]

tbuffer_load_format_xy v[1:2], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_load_format_xy v[1:2], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x79,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_load_format_xy v[1:2], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x80,0x78,0xe9,0x00,0x01,0x01,0x01]

tbuffer_load_format_xyz v[1:3], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_load_format_xyz v[1:3], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x7a,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_load_format_xyz v[1:3], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x79,0xe9,0x00,0x01,0x01,0x01]

tbuffer_load_format_xyzw v[1:4], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_load_format_xyzw v[1:4], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x7b,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_load_format_xyzw v[1:4], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x80,0x79,0xe9,0x00,0x01,0x01,0x01]

tbuffer_store_format_x v1, off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_store_format_x v1, off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x7c,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_store_format_x v1, off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x7a,0xe9,0x00,0x01,0x01,0x01]

tbuffer_store_format_xy v[1:2], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_store_format_xy v[1:2], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x7d,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_store_format_xy v[1:2], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x80,0x7a,0xe9,0x00,0x01,0x01,0x01]

tbuffer_store_format_xyzw v[1:4], off, s[4:7], dfmt:15, nfmt:2, s1
// SICI: tbuffer_store_format_xyzw v[1:4], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x00,0x7f,0xe9,0x00,0x01,0x01,0x01]
// VI:   tbuffer_store_format_xyzw v[1:4], off, s[4:7],  dfmt:15,  nfmt:2, s1 ; encoding: [0x00,0x80,0x7b,0xe9,0x00,0x01,0x01,0x01]

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, nfmt:2, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7],  dfmt:15,  nfmt:2, ttmp1 ; encoding: [0x00,0x00,0x7f,0xe9,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15,  nfmt:2, ttmp1 ; encoding: [0x00,0x80,0x7b,0xe9,0x00,0x01,0x1d,0x71]

// nfmt is optional:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7],  dfmt:15,  nfmt:0, ttmp1 ; encoding: [0x00,0x00,0x7f,0xe8,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15,  nfmt:0, ttmp1 ; encoding: [0x00,0x80,0x7b,0xe8,0x00,0x01,0x1d,0x71]

// dfmt is optional:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], nfmt:2, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:1, nfmt:2, ttmp1 ; encoding: [0x00,0x00,0x0f,0xe9,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:1, nfmt:2, ttmp1 ; encoding: [0x00,0x80,0x0b,0xe9,0x00,0x01,0x1d,0x71]

// nfmt and dfmt can be in either order:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], nfmt:2, dfmt:15, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7],  dfmt:15,  nfmt:2, ttmp1 ; encoding: [0x00,0x00,0x7f,0xe9,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15,  nfmt:2, ttmp1 ; encoding: [0x00,0x80,0x7b,0xe9,0x00,0x01,0x1d,0x71]

// nfmt and dfmt may be omitted:
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x0f,0xe8,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x80,0x0b,0xe8,0x00,0x01,0x1d,0x71]

// Check dfmt/nfmt min values
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:0, nfmt:0, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:0, nfmt:0, ttmp1 ; encoding: [0x00,0x00,0x07,0xe8,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:0, nfmt:0, ttmp1 ; encoding: [0x00,0x80,0x03,0xe8,0x00,0x01,0x1d,0x71]

// Check dfmt/nfmt max values
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, nfmt:7, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, nfmt:7, ttmp1 ; encoding: [0x00,0x00,0xff,0xeb,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, nfmt:7, ttmp1 ; encoding: [0x00,0x80,0xfb,0xeb,0x00,0x01,0x1d,0x71]

// Check default dfmt/nfmt values
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:1, nfmt:0, ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x0f,0xe8,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x80,0x0b,0xe8,0x00,0x01,0x1d,0x71]

// Check that comma separators are optional
tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:15 nfmt:7 ttmp1
// SICI: tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, nfmt:7, ttmp1 ; encoding: [0x00,0x00,0xff,0xeb,0x00,0x01,0x1d,0x71]
// VI:   tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7], dfmt:15, nfmt:7, ttmp1 ; encoding: [0x00,0x80,0xfb,0xeb,0x00,0x01,0x1d,0x71]

//===----------------------------------------------------------------------===//
// Errors handling.
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
// GCN-ERR: error: not a valid operand

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1,, nfmt:1 s0
// GCN-ERR: error: unknown token in expression

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:1,, s0
// GCN-ERR: error: unknown token in expression

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 dfmt:1 s0
// GCN-ERR: error: not a valid operand

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] nfmt:1 nfmt:1 s0
// GCN-ERR: error: not a valid operand

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:1 dfmt:1 s0
// GCN-ERR: error: not a valid operand

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] nfmt:1 dfmt:1 nfmt:1 s0
// GCN-ERR: error: not a valid operand

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1: nfmt:1 s0
// GCN-ERR: error: unknown token in expression

tbuffer_store_format_xyzw v[1:4], off, ttmp[4:7] dfmt:1 nfmt:1: s0
// GCN-ERR: error: unknown token in expression
