// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck -check-prefix=GCN --implicit-check-not=error: %s

s_cbranch_g_fork 100, s[6:7]
// GCN: error: invalid operand for instruction

s_cbranch_g_fork s[6:7], 100
// GCN: error: invalid operand for instruction

s_and_b32 s2, 0x12345678, 0x12345679
// GCN: error: only one unique literal operand is allowed

s_and_b64 s[2:3], 0x12345678, 0x12345679
// GCN: error: only one unique literal operand is allowed
