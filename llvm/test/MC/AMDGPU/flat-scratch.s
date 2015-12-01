// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=SI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=CI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=VI %s

s_mov_b64 flat_scratch, -1
// SI: error: invalid operand for instruction
// CI-NOT: error
// VI-NOT: error

s_mov_b32 flat_scratch_lo, -1
// SI: error: invalid operand for instruction
// CI-NOT: error
// VI-NOT: error

s_mov_b32 flat_scratch_hi, -1
// SI: error: invalid operand for instruction
// CI-NOT: error
// VI-NOT: error


s_mov_b64 flat_scratch_lo, -1
// GCN: error: invalid operand for instruction

s_mov_b64 flat_scratch_hi, -1
// GCN: error: invalid operand for instruction

s_mov_b32 flat_scratch, -1
// GCN: error: invalid operand for instruction
