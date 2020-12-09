// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck -check-prefix=GCN --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=GCN --implicit-check-not=error: %s

exp mrt8 v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp pos4 v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp param32 v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp invalid_target_10 v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp invalid_target_10 v3, v2, v1, v0 done
// GCN: :5: error: invalid exp target

exp invalid_target_11 v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp invalid_target_11 v3, v2, v1, v0 done
// GCN: :5: error: invalid exp target

exp mrt-1 v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp mrtX v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp pos-1 v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp posX v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp param-1 v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp paramX v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp invalid_target_-1 v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp invalid_target_X v3, v2, v1, v0
// GCN: :5: error: invalid exp target

exp 0 v3, v2, v1, v0
// GCN: :5: error: invalid operand for instruction

exp , v3, v2, v1, v0
// GCN: :5: error: unknown token in expression

exp
// GCN: :1: error: too few operands for instruction

exp mrt0 s0, v0, v0, v0
// GCN: 10: error: invalid operand for instruction

exp mrt0 v0, s0, v0, v0
// GCN: 14: error: invalid operand for instruction

exp mrt0 v0, v0, s0, v0
// GCN: 18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, s0
// GCN: 22: error: invalid operand for instruction

exp mrt0 v[0:1], v0, v0, v0
// GCN: 10: error: invalid operand for instruction

exp mrt0 v0, v[0:1], v0, v0
// GCN: 14: error: invalid operand for instruction

exp mrt0 v0, v0, v[0:1], v0
// GCN: 18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, v[0:1]
// GCN: 22: error: invalid operand for instruction

exp mrt0 1.0, v0, v0, v0
// GCN: 10: error: invalid operand for instruction

exp mrt0 v0, 1.0, v0, v0
// GCN: 14: error: invalid operand for instruction

exp mrt0 v0, v0, 1.0, v0
// GCN: 18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, 1.0
// GCN: 22: error: invalid operand for instruction

exp mrt0 7, v0, v0, v0
// GCN: 10: error: invalid operand for instruction

exp mrt0 v0, 7, v0, v0
// GCN: 14: error: invalid operand for instruction

exp mrt0 v0, v0, 7, v0
// GCN: 18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, 7
// GCN: 22: error: invalid operand for instruction

exp mrt0 0x12345678, v0, v0, v0
// GCN: 10: error: invalid operand for instruction

exp mrt0 v0, 0x12345678, v0, v0
// GCN: 14: error: invalid operand for instruction

exp mrt0 v0, v0, 0x12345678, v0
// GCN: 18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, 0x12345678
// GCN: 22: error: invalid operand for instruction
