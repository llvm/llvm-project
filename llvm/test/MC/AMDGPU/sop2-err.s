// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck -check-prefix=GCN %s

s_cbranch_g_fork 100, s[6:7]
// GCN: error: invalid operand for instruction

s_cbranch_g_fork s[6:7], 100
// GCN: error: invalid operand for instruction
