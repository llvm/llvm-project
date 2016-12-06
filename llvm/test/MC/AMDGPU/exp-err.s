// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck -check-prefix=GCN %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck -check-prefix=GCN %s

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
// GCN: :5: error: failed parsing operand

exp mrtX v3, v2, v1, v0
// GCN: :5: error: failed parsing operand

exp pos-1 v3, v2, v1, v0
// GCN: :5: error: failed parsing operand

exp posX v3, v2, v1, v0
// GCN: :5: error: failed parsing operand

exp param-1 v3, v2, v1, v0
// GCN: :5: error: failed parsing operand

exp paramX v3, v2, v1, v0
// GCN: :5: error: failed parsing operand

exp invalid_target_-1 v3, v2, v1, v0
// GCN: :5: error: failed parsing operand

exp invalid_target_X v3, v2, v1, v0
// GCN: :5: error: failed parsing operand

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
