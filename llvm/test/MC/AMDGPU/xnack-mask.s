// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck -check-prefix=NOSICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii -show-encoding %s 2>&1 | FileCheck -check-prefix=NOSICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck -check-prefix=NOSICIVI %s

// RUN: not llvm-mc -arch=amdgcn -mcpu=stoney -show-encoding %s 2>&1 | FileCheck -check-prefix=XNACKERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=stoney -show-encoding %s | FileCheck -check-prefix=XNACK %s

s_mov_b64 xnack_mask, -1
// NOSICIVI: error: not a valid operand.
// XNACK:    s_mov_b64 xnack_mask, -1 ; encoding: [0xc1,0x01,0xe8,0xbe]

s_mov_b32 xnack_mask_lo, -1
// NOSICIVI: error: not a valid operand.
// XNACK:    s_mov_b32 xnack_mask_lo, -1 ; encoding: [0xc1,0x00,0xe8,0xbe]

s_mov_b32 xnack_mask_hi, -1
// NOSICIVI: error: not a valid operand.
// XNACK:    s_mov_b32 xnack_mask_hi, -1 ; encoding: [0xc1,0x00,0xe9,0xbe]

s_mov_b32 xnack_mask, -1
// NOSICIVI: error: not a valid operand.
// XNACKERR: error: invalid operand for instruction

s_mov_b64 xnack_mask_lo, -1
// NOSICIVI: error: not a valid operand.
// XNACKERR: error: invalid operand for instruction

s_mov_b64 xnack_mask_hi, -1
// NOSICIVI: error: not a valid operand.
// XNACKERR: error: invalid operand for instruction
