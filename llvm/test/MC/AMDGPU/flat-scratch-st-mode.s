// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga 2>&1 %s | FileCheck -check-prefix=VI-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 2>&1 %s | FileCheck -check-prefix=GFX9_10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 2>&1 %s | FileCheck --check-prefixes=GFX9_10-ERR --implicit-check-not=error: %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1030 -show-encoding %s | FileCheck --check-prefixes=GFX1030 %s

scratch_load_ubyte v1, off, off
// GFX1030: encoding: [0x00,0x40,0x20,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_sbyte v1, off, off
// GFX1030: encoding: [0x00,0x40,0x24,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_ushort v1, off, off
// GFX1030: encoding: [0x00,0x40,0x28,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_sshort v1, off, off
// GFX1030: encoding: [0x00,0x40,0x2c,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_dword v1, off, off
// GFX1030: encoding: [0x00,0x40,0x30,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_dwordx2 v[1:2], off, off
// GFX1030: encoding: [0x00,0x40,0x34,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_dwordx3 v[1:3], off, off
// GFX1030: encoding: [0x00,0x40,0x3c,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_dwordx4 v[1:4], off, off
// GFX1030: encoding: [0x00,0x40,0x38,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_dword v1, off, off offset:2047
// GFX1030: scratch_load_dword v1, off, off offset:2047 ; encoding: [0xff,0x47,0x30,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_byte off, v2, off
// GFX1030: encoding: [0x00,0x40,0x60,0xdc,0x00,0x02,0x7f,0x00]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_short off, v2, off
// GFX1030: encoding: [0x00,0x40,0x68,0xdc,0x00,0x02,0x7f,0x00]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_dword off, v2, off
// GFX1030: encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x7f,0x00]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_dwordx2 off, v[2:3], off
// GFX1030: encoding: [0x00,0x40,0x74,0xdc,0x00,0x02,0x7f,0x00]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_dwordx3 off, v[2:4], off
// GFX1030: encoding: [0x00,0x40,0x7c,0xdc,0x00,0x02,0x7f,0x00]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_dwordx4 off, v[2:5], off
// GFX1030: encoding: [0x00,0x40,0x78,0xdc,0x00,0x02,0x7f,0x00]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_dword off, v2, off offset:2047
// GFX1030: scratch_store_dword off, v2, off offset:2047 ; encoding: [0xff,0x47,0x70,0xdc,0x00,0x02,0x7f,0x00]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_ubyte_d16 v1, off, off
// GFX1030: encoding: [0x00,0x40,0x80,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_ubyte_d16_hi v1, off, off
// GFX1030: encoding: [0x00,0x40,0x84,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_sbyte_d16 v1, off, off
// GFX1030: encoding: [0x00,0x40,0x88,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_sbyte_d16_hi v1, off, off
// GFX1030: encoding: [0x00,0x40,0x8c,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_short_d16 v1, off, off
// GFX1030: encoding: [0x00,0x40,0x90,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_load_short_d16_hi v1, off, off
// GFX1030: encoding: [0x00,0x40,0x94,0xdc,0x00,0x00,0x7f,0x01]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

scratch_store_byte_d16_hi off, v2, off
// GFX1030: encoding: [0x00,0x40,0x64,0xdc,0x00,0x02,0x7f,0x00]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU


scratch_store_short_d16_hi off, v2, off
// GFX1030: encoding: [0x00,0x40,0x6c,0xdc,0x00,0x02,0x7f,0x00]
// GFX9_10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU
