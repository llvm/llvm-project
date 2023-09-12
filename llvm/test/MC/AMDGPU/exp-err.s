// RUN: not llvm-mc -triple=amdgcn %s 2>&1 | FileCheck -check-prefixes=GCN,GFX68 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefixes=GCN,GFX68 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck -check-prefixes=GCN,GFX11 --implicit-check-not=error: %s

exp mrt8 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp pos4 v3, v2, v1, v0
// GFX68: :[[@LINE-1]]:5: error: exp target is not supported on this GPU

exp pos5 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp param32 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp invalid_target_10 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp invalid_target_10 v3, v2, v1, v0 done
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp invalid_target_11 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp invalid_target_11 v3, v2, v1, v0 done
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp mrt-1 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp mrtX v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp pos-1 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp posX v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp param-1 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp paramX v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp invalid_target_-1 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp invalid_target_X v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid exp target

exp 0 v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: invalid operand for instruction

exp , v3, v2, v1, v0
// GCN: :[[@LINE-1]]:5: error: unknown token in expression

exp
// GCN: :[[@LINE-1]]:1: error: too few operands for instruction

exp mrt0 s0, v0, v0, v0
// GCN: :[[@LINE-1]]:10: error: invalid operand for instruction

exp mrt0 v0, s0, v0, v0
// GCN: :[[@LINE-1]]:14: error: invalid operand for instruction

exp mrt0 v0, v0, s0, v0
// GCN: :[[@LINE-1]]:18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, s0
// GCN: :[[@LINE-1]]:22: error: invalid operand for instruction

exp mrt0 v[0:1], v0, v0, v0
// GCN: :[[@LINE-1]]:10: error: invalid operand for instruction

exp mrt0 v0, v[0:1], v0, v0
// GCN: :[[@LINE-1]]:14: error: invalid operand for instruction

exp mrt0 v0, v0, v[0:1], v0
// GCN: :[[@LINE-1]]:18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, v[0:1]
// GCN: :[[@LINE-1]]:22: error: invalid operand for instruction

exp mrt0 1.0, v0, v0, v0
// GCN: :[[@LINE-1]]:10: error: invalid operand for instruction

exp mrt0 v0, 1.0, v0, v0
// GCN: :[[@LINE-1]]:14: error: invalid operand for instruction

exp mrt0 v0, v0, 1.0, v0
// GCN: :[[@LINE-1]]:18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, 1.0
// GCN: :[[@LINE-1]]:22: error: invalid operand for instruction

exp mrt0 7, v0, v0, v0
// GCN: :[[@LINE-1]]:10: error: invalid operand for instruction

exp mrt0 v0, 7, v0, v0
// GCN: :[[@LINE-1]]:14: error: invalid operand for instruction

exp mrt0 v0, v0, 7, v0
// GCN: :[[@LINE-1]]:18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, 7
// GCN: :[[@LINE-1]]:22: error: invalid operand for instruction

exp mrt0 0x12345678, v0, v0, v0
// GCN: :[[@LINE-1]]:10: error: invalid operand for instruction

exp mrt0 v0, 0x12345678, v0, v0
// GCN: :[[@LINE-1]]:14: error: invalid operand for instruction

exp mrt0 v0, v0, 0x12345678, v0
// GCN: :[[@LINE-1]]:18: error: invalid operand for instruction

exp mrt0 v0, v0, v0, 0x12345678
// GCN: :[[@LINE-1]]:22: error: invalid operand for instruction

exp null v4, v3, v2, v1
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: exp target is not supported on this GPU

exp param0 v4, v3, v2, v1
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: exp target is not supported on this GPU

exp param31 v4, v3, v2, v1
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: exp target is not supported on this GPU

exp mrt0 v4, v3, v2, v1 vm
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

exp mrtz, v3, v3, off, off compr
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
