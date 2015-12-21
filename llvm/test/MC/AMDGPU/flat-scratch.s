// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck -check-prefix=SI -check-prefix=GCN %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii -show-encoding %s | FileCheck -check-prefix=CI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s  | FileCheck -check-prefix=VI %s

// Add a different RUN line for the failing checks, because when stderr and stdout are mixed the
// order things are printed is not deterministic.
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii -show-encoding %s 2>&1 | FileCheck -check-prefix=GCN %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck -check-prefix=GCN %s

s_mov_b64 flat_scratch, -1
// SI: error: invalid operand for instruction
// CI: s_mov_b64 flat_scratch, -1 ; encoding: [0xc1,0x04,0xe8,0xbe]
// VI: s_mov_b64 flat_scratch, -1 ; encoding: [0xc1,0x01,0xe6,0xbe]

s_mov_b32 flat_scratch_lo, -1
// SI: error: invalid operand for instruction
// CI: s_mov_b32 flat_scratch_lo, -1 ; encoding: [0xc1,0x03,0xe8,0xbe]
// VI: s_mov_b32 flat_scratch_lo, -1 ; encoding: [0xc1,0x00,0xe6,0xbe]

s_mov_b32 flat_scratch_hi, -1
// SI: error: invalid operand for instruction
// CI: s_mov_b32 flat_scratch_hi, -1 ; encoding: [0xc1,0x03,0xe9,0xbe]
// VI: s_mov_b32 flat_scratch_hi, -1 ; encoding: [0xc1,0x00,0xe7,0xbe]


s_mov_b64 flat_scratch_lo, -1
// GCN: error: invalid operand for instruction

s_mov_b64 flat_scratch_hi, -1
// GCN: error: invalid operand for instruction

s_mov_b32 flat_scratch, -1
// GCN: error: invalid operand for instruction
