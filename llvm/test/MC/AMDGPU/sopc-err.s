// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=VI --implicit-check-not=error: %s

s_set_gpr_idx_on s0, s1
// VI: error: expected absolute expression

s_set_gpr_idx_on s0, 16
// VI: error: invalid immediate: only 4-bit values are legal

s_set_gpr_idx_on s0, -1
// VI: error: invalid immediate: only 4-bit values are legal

s_set_gpr_idx_on s0, gpr_idx
// VI: error: expected absolute expression

s_set_gpr_idx_on s0, gpr_idx(
// VI: error: expected a VGPR index mode or a closing parenthesis

s_set_gpr_idx_on s0, gpr_idx(X)
// VI: error: expected a VGPR index mode

s_set_gpr_idx_on s0, gpr_idx(SRC0,DST,SRC1,DST)
// VI: error: duplicate VGPR index mode

s_set_gpr_idx_on s0, gpr_idx(DST
// VI: error: expected a comma or a closing parenthesis

s_set_gpr_idx_on s0, gpr_idx(SRC0,
// VI: error: expected a VGPR index mode

s_cmp_eq_i32 0x12345678, 0x12345679
// GCN: error: only one literal operand is allowed

s_cmp_eq_u64 0x12345678, 0x12345679
// GCN: error: only one literal operand is allowed
