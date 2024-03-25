// RUN: not llvm-mc -triple=amdgcn %s 2>&1 | FileCheck -check-prefix=GCN --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=GCN --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck --check-prefixes=GCN,VI --implicit-check-not=error: %s

s_mov_b32 v1, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s1, v0
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b32 s[1:2], s0
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment

s_mov_b32 s0, s[1:2]
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment

s_mov_b32 s220, s0
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_mov_b32 s0, s220
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_mov_b64 s1, s[0:1]
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b64 s[0:1], s1
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

// FIXME: This shoudl probably say failed to parse.
s_mov_b32 s
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// Out of range register

s_mov_b32 s102, 1
// VI: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU

s_mov_b32 s103, 1
// VI: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU

s_mov_b64 s[102:103], -1
// VI: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU

s_setpc_b64 0
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
