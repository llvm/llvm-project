// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX1200-ERR %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefix=GFX1250 %s

s_mov_b32 s0, src_flat_scratch_base_lo
// GFX1200-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: src_flat_scratch_base_lo register not available on this GPU
// GFX1250: encoding: [0xe6,0x00,0x80,0xbe]

s_mov_b32 s0, src_flat_scratch_base_hi
// GFX1200-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: src_flat_scratch_base_hi register not available on this GPU
// GFX1250: encoding: [0xe7,0x00,0x80,0xbe]

s_mov_b64 s[0:1], src_flat_scratch_base_lo
// GFX1200-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: src_flat_scratch_base_lo register not available on this GPU
// GFX1250: encoding: [0xe6,0x01,0x80,0xbe]

s_mov_b64 s[0:1], shared_base
// GFX1250: encoding: [0xeb,0x01,0x80,0xbe]

s_mov_b64 s[0:1], src_shared_base
// GFX1250: encoding: [0xeb,0x01,0x80,0xbe]

s_mov_b64 s[0:1], shared_limit
// GFX1250: encoding: [0xec,0x01,0x80,0xbe]

s_mov_b64 s[0:1], src_shared_limit
// GFX1250: encoding: [0xec,0x01,0x80,0xbe]

s_getreg_b32 s1, hwreg(33)
// GFX1250: encoding: [0x21,0xf8,0x81,0xb8]

s_getreg_b32 s1, hwreg(HW_REG_XNACK_STATE_PRIV)
// GFX1200-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX1250: encoding: [0x21,0xf8,0x81,0xb8]

s_getreg_b32 s1, hwreg(34)
// GFX1250: encoding: [0x22,0xf8,0x81,0xb8]

s_getreg_b32 s1, hwreg(HW_REG_XNACK_MASK)
// GFX1200-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX1250: encoding: [0x22,0xf8,0x81,0xb8]

s_setreg_b32 hwreg(33), s1
// GFX1250: encoding: [0x21,0xf8,0x01,0xb9]

s_setreg_b32 hwreg(HW_REG_XNACK_STATE_PRIV), s1
// GFX1200-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX1250: encoding: [0x21,0xf8,0x01,0xb9]

s_setreg_b32 hwreg(34), s1
// GFX1250: encoding: [0x22,0xf8,0x01,0xb9]

s_setreg_b32 hwreg(HW_REG_XNACK_MASK), s1
// GFX1200-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX1250: encoding: [0x22,0xf8,0x01,0xb9]

s_setreg_b32 hwreg(HW_REG_IB_STS2), s1
// GFX1200-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX1250: encoding: [0x1c,0xf8,0x01,0xb9]
