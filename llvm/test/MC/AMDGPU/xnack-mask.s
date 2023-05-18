// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=NOSICIVI10 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=hawaii %s 2>&1 | FileCheck -check-prefix=NOSICIVI10 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOSICIVI10 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1001 -mattr=-xnack %s 2>&1 | FileCheck -check-prefix=NOSICIVI10 --implicit-check-not=error: %s

// RUN: not llvm-mc -triple=amdgcn -mcpu=stoney -mattr=+xnack %s 2>&1 | FileCheck -check-prefix=XNACKERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=stoney -mattr=+xnack -show-encoding %s | FileCheck -check-prefix=XNACK %s

s_mov_b64 xnack_mask, -1
// NOSICIVI10: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU
// XNACK:    s_mov_b64 xnack_mask, -1 ; encoding: [0xc1,0x01,0xe8,0xbe]

s_mov_b32 xnack_mask_lo, -1
// NOSICIVI10: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU
// XNACK:    s_mov_b32 xnack_mask_lo, -1 ; encoding: [0xc1,0x00,0xe8,0xbe]

s_mov_b32 xnack_mask_hi, -1
// NOSICIVI10: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU
// XNACK:    s_mov_b32 xnack_mask_hi, -1 ; encoding: [0xc1,0x00,0xe9,0xbe]

s_mov_b32 xnack_mask, -1
// NOSICIVI10: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU
// XNACKERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b64 xnack_mask_lo, -1
// NOSICIVI10: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU
// XNACKERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction

s_mov_b64 xnack_mask_hi, -1
// NOSICIVI10: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU
// XNACKERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction
