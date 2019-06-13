// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding < %s 2>&1 | FileCheck -check-prefix=GFX89 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding < %s 2>&1 | FileCheck -check-prefix=GFX89 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding < %s 2>&1 | FileCheck -check-prefix=GFX10 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding < %s 2>&1 | FileCheck -check-prefix=GFX89-ERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding < %s 2>&1 | FileCheck -check-prefix=GFX89-ERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding < %s 2>&1 | FileCheck -check-prefix=GFX10-ERR %s

v_mov_b32_dpp v0, v1 row_share:1 row_mask:0x1 bank_mask:0x1
// GFX89-ERR: not a valid operand.
// GFX10:     v_mov_b32_dpp v0, v1  row_share:1 row_mask:0x1 bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x51,0x01,0x11]

v_mov_b32_dpp v0, v1 row_xmask:1 row_mask:0x1 bank_mask:0x1
// GFX89-ERR: not a valid operand.
// GFX10:     v_mov_b32_dpp v0, v1  row_xmask:1 row_mask:0x1 bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x61,0x01,0x11]

v_mov_b32_dpp v0, v1 wave_shl:1 row_mask:0x1 bank_mask:0x1
// GFX89:     v0, v1 wave_shl:1 row_mask:0x1 bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x30,0x01,0x11]
// GFX10-ERR: not a valid operand.

v_mov_b32_dpp v0, v1 wave_shr:1 row_mask:0x1 bank_mask:0x1
// GFX89:     v0, v1 wave_shr:1 row_mask:0x1 bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x38,0x01,0x11]
// GFX10-ERR: not a valid operand.

v_mov_b32_dpp v0, v1 wave_rol:1 row_mask:0x1 bank_mask:0x1
// GFX89:     v0, v1 wave_rol:1 row_mask:0x1 bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x34,0x01,0x11]
// GFX10-ERR: not a valid operand.

v_mov_b32_dpp v0, v1 wave_ror:1 row_mask:0x1 bank_mask:0x1
// GFX89:     v0, v1 wave_ror:1 row_mask:0x1 bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x3c,0x01,0x11]
// GFX10-ERR: not a valid operand.

v_mov_b32_dpp v0, v1 row_bcast:15 row_mask:0x1 bank_mask:0x1
// GFX89:     v0, v1 row_bcast:15 row_mask:0x1 bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x42,0x01,0x11]
// GFX10-ERR: not a valid operand.

v_mov_b32_dpp v0, v1 row_bcast:31 row_mask:0x1 bank_mask:0x1
// GFX89:     v0, v1 row_bcast:31 row_mask:0x1 bank_mask:0x1 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x43,0x01,0x11]
// GFX10-ERR: not a valid operand.
