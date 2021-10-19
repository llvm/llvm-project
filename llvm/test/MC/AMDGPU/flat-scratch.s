// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=NOSI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii %s 2>&1 | FileCheck -check-prefix=NOCI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOVI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii -show-encoding %s | FileCheck -check-prefix=CI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s  | FileCheck -check-prefix=VI %s

s_mov_b64 flat_scratch, -1
// NOSI: error: register not available on this GPU
// CI: s_mov_b64 flat_scratch, -1 ; encoding: [0xc1,0x04,0xe8,0xbe]
// VI: s_mov_b64 flat_scratch, -1 ; encoding: [0xc1,0x01,0xe6,0xbe]

s_mov_b32 flat_scratch_lo, -1
// NOSI: error: register not available on this GPU
// CI: s_mov_b32 flat_scratch_lo, -1 ; encoding: [0xc1,0x03,0xe8,0xbe]
// VI: s_mov_b32 flat_scratch_lo, -1 ; encoding: [0xc1,0x00,0xe6,0xbe]

s_mov_b32 flat_scratch_hi, -1
// NOSI: error: register not available on this GPU
// CI: s_mov_b32 flat_scratch_hi, -1 ; encoding: [0xc1,0x03,0xe9,0xbe]
// VI: s_mov_b32 flat_scratch_hi, -1 ; encoding: [0xc1,0x00,0xe7,0xbe]


s_mov_b64 flat_scratch_lo, -1
// NOSI: error: register not available on this GPU
// NOCI: error: invalid operand for instruction
// NOVI: error: invalid operand for instruction

s_mov_b64 flat_scratch_hi, -1
// NOSI: error: register not available on this GPU
// NOCI: error: invalid operand for instruction
// NOVI: error: invalid operand for instruction

s_mov_b32 flat_scratch, -1
// NOSI: error: register not available on this GPU
// NOCI: error: invalid operand for instruction
// NOVI: error: invalid operand for instruction
