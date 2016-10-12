// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=VI %s

s_set_gpr_idx_on s0, s1
// GCN: error: invalid operand for instruction

s_set_gpr_idx_on s0, 16
// GCN: error: invalid operand for instruction

s_set_gpr_idx_on s0, -1
// GCN: error: invalid operand for instruction
