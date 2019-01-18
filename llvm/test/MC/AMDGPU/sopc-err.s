// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=VI %s

s_set_gpr_idx_on s0, s1
// GCN: error: invalid operand for instruction

s_set_gpr_idx_on s0, 16
// GCN: error: invalid operand for instruction

s_set_gpr_idx_on s0, -1
// GCN: error: invalid operand for instruction

s_cmp_eq_i32 0x12345678, 0x12345679
// GCN: error: only one literal operand is allowed

s_cmp_eq_u64 0x12345678, 0x12345679
// GCN: error: only one literal operand is allowed
