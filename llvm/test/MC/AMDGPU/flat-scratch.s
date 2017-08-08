// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck -check-prefix=NOSI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii -show-encoding %s 2>&1 | FileCheck -check-prefix=NOCI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck -check-prefix=NOVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii -show-encoding %s | FileCheck -check-prefix=CI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s  | FileCheck -check-prefix=VI %s

s_mov_b64 flat_scratch, -1
// NOSI: error: not a valid operand.
// CI: s_mov_b64 flat_scratch, -1 ; encoding: [0xc1,0x04,0xe8,0xbe]
// VI: s_mov_b64 flat_scratch, -1 ; encoding: [0xc1,0x01,0xe6,0xbe]

s_mov_b32 flat_scratch_lo, -1
// NOSI: error: not a valid operand.
// CI: s_mov_b32 flat_scratch_lo, -1 ; encoding: [0xc1,0x03,0xe8,0xbe]
// VI: s_mov_b32 flat_scratch_lo, -1 ; encoding: [0xc1,0x00,0xe6,0xbe]

s_mov_b32 flat_scratch_hi, -1
// NOSI: error: not a valid operand.
// CI: s_mov_b32 flat_scratch_hi, -1 ; encoding: [0xc1,0x03,0xe9,0xbe]
// VI: s_mov_b32 flat_scratch_hi, -1 ; encoding: [0xc1,0x00,0xe7,0xbe]


s_mov_b64 flat_scratch_lo, -1
// NOSI: error: not a valid operand.
// NOCI: error: invalid operand for instruction
// NOVI: error: invalid operand for instruction

s_mov_b64 flat_scratch_hi, -1
// NOSI: error: not a valid operand.
// NOCI: error: invalid operand for instruction
// NOVI: error: invalid operand for instruction

s_mov_b32 flat_scratch, -1
// NOSI: error: not a valid operand.
// NOCI: error: invalid operand for instruction
// NOVI: error: invalid operand for instruction
