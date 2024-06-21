// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s | FileCheck --check-prefix=GFX12 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: %s

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15]
// GFX12: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] ; encoding: [0x08,0x40,0x40,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[1,0,0]
// GFX12: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[1,0,0] ; encoding: [0x08,0x40,0x40,0xcc,0x00,0x09,0x22,0x3c]

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[0,1,0]
// GFX12: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[0,1,0] ; encoding: [0x08,0x40,0x40,0xcc,0x00,0x09,0x22,0x5c]

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[0,0,1] ; encoding: [0x08,0x40,0x40,0xcc,0x00,0x09,0x22,0x9c]

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[1,0,0]
// GFX12: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[1,0,0] ; encoding: [0x08,0x41,0x40,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[0,1,0]
// GFX12: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[0,1,0] ; encoding: [0x08,0x42,0x40,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[0,0,1] ; encoding: [0x08,0x44,0x40,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f32_16x16x16_f16 v[8:15], s[0:3], v[4:7], v[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], s[4:7], v[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], s[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[8:15], 1.0, v[4:7], v[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], 1.0, v[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], 1.0
// GFX12: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], 1.0 ; encoding: [0x08,0x40,0x40,0xcc,0x00,0x09,0xca,0x1b]

v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], 1
// GFX12: v_wmma_f32_16x16x16_f16 v[8:15], v[0:3], v[4:7], 1 ; encoding: [0x08,0x40,0x40,0xcc,0x00,0x09,0x06,0x1a]



v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15]
// GFX12: v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] ; encoding: [0x08,0x40,0x41,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[1,0,0]
// GFX12: v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[1,0,0] ; encoding: [0x08,0x40,0x41,0xcc,0x00,0x09,0x22,0x3c]

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[0,1,0]
// GFX12: v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[0,1,0] ; encoding: [0x08,0x40,0x41,0xcc,0x00,0x09,0x22,0x5c]

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_lo:[0,0,1] ; encoding: [0x08,0x40,0x41,0xcc,0x00,0x09,0x22,0x9c]

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[1,0,0]
// GFX12: v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[1,0,0] ; encoding: [0x08,0x41,0x41,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[0,1,0]
// GFX12: v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[0,1,0] ; encoding: [0x08,0x42,0x41,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], v[8:15] neg_hi:[0,0,1] ; encoding: [0x08,0x44,0x41,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f32_16x16x16_bf16 v[8:15], s[0:3], v[4:7], v[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], s[4:7], v[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], s[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[8:15], 1.0, v[4:7], v[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], 1.0, v[8:15]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], 1.0
// GFX12: v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], 1.0 ; encoding: [0x08,0x40,0x41,0xcc,0x00,0x09,0xca,0x1b]

v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], 1
// GFX12: v_wmma_f32_16x16x16_bf16 v[8:15], v[0:3], v[4:7], 1 ; encoding: [0x08,0x40,0x41,0xcc,0x00,0x09,0x06,0x1a]



v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11]
// GFX12: v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] ; encoding: [0x08,0x40,0x42,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[1,0,0]
// GFX12: v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[1,0,0] ; encoding: [0x08,0x40,0x42,0xcc,0x00,0x09,0x22,0x3c]

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[0,1,0]
// GFX12: v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[0,1,0] ; encoding: [0x08,0x40,0x42,0xcc,0x00,0x09,0x22,0x5c]

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[0,0,1]
// GFX12: v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[0,0,1] ; encoding: [0x08,0x40,0x42,0xcc,0x00,0x09,0x22,0x9c]

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[1,0,0]
// GFX12: v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[1,0,0] ; encoding: [0x08,0x41,0x42,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[0,1,0]
// GFX12: v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[0,1,0] ; encoding: [0x08,0x42,0x42,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[0,0,1]
// GFX12: v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[0,0,1] ; encoding: [0x08,0x44,0x42,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_f16_16x16x16_f16 v[8:11], s[0:3], v[4:7], v[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], s[4:7], v[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], s[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[8:11], 1.0, v[4:7], v[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], 1.0, v[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], 1.0
// GFX12: v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], 1.0 ; encoding: [0x08,0x40,0x42,0xcc,0x00,0x09,0xca,0x1b]

v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], 1
// GFX12: v_wmma_f16_16x16x16_f16 v[8:11], v[0:3], v[4:7], 1 ; encoding: [0x08,0x40,0x42,0xcc,0x00,0x09,0x06,0x1a]



v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] ; encoding: [0x08,0x40,0x43,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[1,0,0]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[1,0,0] ; encoding: [0x08,0x40,0x43,0xcc,0x00,0x09,0x22,0x3c]

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[0,1,0]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[0,1,0] ; encoding: [0x08,0x40,0x43,0xcc,0x00,0x09,0x22,0x5c]

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[0,0,1]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_lo:[0,0,1] ; encoding: [0x08,0x40,0x43,0xcc,0x00,0x09,0x22,0x9c]

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[1,0,0]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[1,0,0] ; encoding: [0x08,0x41,0x43,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[0,1,0]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[0,1,0] ; encoding: [0x08,0x42,0x43,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[0,0,1]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], v[8:11] neg_hi:[0,0,1] ; encoding: [0x08,0x44,0x43,0xcc,0x00,0x09,0x22,0x1c]

v_wmma_bf16_16x16x16_bf16 v[8:11], s[0:3], v[4:7], v[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], s[4:7], v[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], s[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[8:11], 1.0, v[4:7], v[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], 1.0, v[8:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], 1.0
// GFX12: v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], 1.0 ; encoding: [0x08,0x40,0x43,0xcc,0x00,0x09,0xca,0x1b]

v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], 1
// GFX12: v_wmma_bf16_16x16x16_bf16 v[8:11], v[0:3], v[4:7], 1 ; encoding: [0x08,0x40,0x43,0xcc,0x00,0x09,0x06,0x1a]



v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11]
// GFX12: v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] ; encoding: [0x04,0x40,0x44,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] clamp
// GFX12: v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] clamp ; encoding: [0x04,0xc0,0x44,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0]
// GFX12: v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0] ; encoding: [0x04,0x40,0x44,0xcc,0x00,0x05,0x12,0x3c]

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0]
// GFX12: v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0] ; encoding: [0x04,0x40,0x44,0xcc,0x00,0x05,0x12,0x5c]

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[4:11], s[0:1], v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], s[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], s[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[4:11], 1, v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], 1, v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], 1
// GFX12: v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x44,0xcc,0x00,0x05,0x06,0x1a]

v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_i32_16x16x16_iu8 v[4:11], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x44,0xcc,0x00,0x05,0xca,0x1b]



v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9]
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] clamp
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] clamp ; encoding: [0x02,0xc0,0x45,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] neg_lo:[1,0,0]
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] neg_lo:[1,0,0] ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0x0a,0x3c]

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] neg_lo:[0,1,0]
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] neg_lo:[0,1,0] ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0x0a,0x5c]

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, v[2:9] neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:9], s0, v1, v[2:9]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:9], v0, s1, v[2:9]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, s[0:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:9], 1, v1, v[2:9]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:9], v0, 1, v[2:9]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, 1
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, 1 ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0x06,0x1a]

v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, 1.0
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:9], v0, v1, 1.0 ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0xca,0x1b]



v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11]
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] ; encoding: [0x04,0x40,0x46,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x46,0xcc,0x00,0x05,0x12,0x9c]

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1] ; encoding: [0x04,0x44,0x46,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], s[0:1], v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], s[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], s[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], 1.0, v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], 1.0, v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x46,0xcc,0x00,0x05,0xca,0x1b]

v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], 1
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[4:11], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x46,0xcc,0x00,0x05,0x06,0x1a]



v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11]
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] ; encoding: [0x04,0x40,0x48,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x48,0xcc,0x00,0x05,0x12,0x9c]

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1] ; encoding: [0x04,0x44,0x48,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], s[0:1], v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], s[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], s[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], 1.0, v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], 1.0, v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x48,0xcc,0x00,0x05,0xca,0x1b]

v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], 1
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[4:11], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x48,0xcc,0x00,0x05,0x06,0x1a]



v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11]
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] ; encoding: [0x04,0x40,0x47,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x47,0xcc,0x00,0x05,0x12,0x9c]

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1] ; encoding: [0x04,0x44,0x47,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], s[0:1], v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], s[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], s[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], 1.0, v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], 1.0, v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x47,0xcc,0x00,0x05,0xca,0x1b]

v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], 1
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[4:11], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x47,0xcc,0x00,0x05,0x06,0x1a]



v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11]
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] ; encoding: [0x04,0x40,0x49,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x49,0xcc,0x00,0x05,0x12,0x9c]

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1] ; encoding: [0x04,0x44,0x49,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], s[0:1], v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], s[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], s[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], 1.0, v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], 1.0, v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x49,0xcc,0x00,0x05,0xca,0x1b]

v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], 1
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[4:11], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x49,0xcc,0x00,0x05,0x06,0x1a]



v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11]
// GFX12: v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] ; encoding: [0x04,0x40,0x4a,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] clamp
// GFX12: v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] clamp ; encoding: [0x04,0xc0,0x4a,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0]
// GFX12: v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0] ; encoding: [0x04,0x40,0x4a,0xcc,0x00,0x05,0x12,0x3c]

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0]
// GFX12: v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0] ; encoding: [0x04,0x40,0x4a,0xcc,0x00,0x05,0x12,0x5c]

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[4:11], s[0:1], v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], s[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], s[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[4:11], 1, v[2:3], v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], 1, v[4:11]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], 1
// GFX12: v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x4a,0xcc,0x00,0x05,0x06,0x1a]

v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_i32_16x16x32_iu4 v[4:11], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x4a,0xcc,0x00,0x05,0xca,0x1b]



v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20
// GFX12: v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 ; encoding: [0x0c,0x40,0x50,0xcc,0x00,0x09,0x52,0x1c]

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 index_key:1
// GFX12: v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 index_key:1 ; encoding: [0x0c,0x48,0x50,0xcc,0x00,0x09,0x52,0x1c]

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[1,0,0]
// GFX12: v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[1,0,0] ; encoding: [0x0c,0x40,0x50,0xcc,0x00,0x09,0x52,0x3c]

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[0,1,0]
// GFX12: v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[0,1,0] ; encoding: [0x0c,0x40,0x50,0xcc,0x00,0x09,0x52,0x5c]

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[1,0,0]
// GFX12: v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[1,0,0] ; encoding: [0x0c,0x41,0x50,0xcc,0x00,0x09,0x52,0x1c]

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[0,1,0]
// GFX12: v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[0,1,0] ; encoding: [0x0c,0x42,0x50,0xcc,0x00,0x09,0x52,0x1c]

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_swmmac_f32_16x16x32_f16 v[12:19], s[0:3], v[4:11], v20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], s[4:11], v20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], s20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[12:19], 1.0, v[4:11], v20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], 1.0, v20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[12:19], v[0:3], v[4:11], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20
// GFX12: v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 ; encoding: [0x0c,0x40,0x51,0xcc,0x00,0x09,0x52,0x1c]

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 index_key:1
// GFX12: v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 index_key:1 ; encoding: [0x0c,0x48,0x51,0xcc,0x00,0x09,0x52,0x1c]

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[1,0,0]
// GFX12: v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[1,0,0] ; encoding: [0x0c,0x40,0x51,0xcc,0x00,0x09,0x52,0x3c]

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[0,1,0]
// GFX12: v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[0,1,0] ; encoding: [0x0c,0x40,0x51,0xcc,0x00,0x09,0x52,0x5c]

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[1,0,0]
// GFX12: v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[1,0,0] ; encoding: [0x0c,0x41,0x51,0xcc,0x00,0x09,0x52,0x1c]

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[0,1,0]
// GFX12: v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[0,1,0] ; encoding: [0x0c,0x42,0x51,0xcc,0x00,0x09,0x52,0x1c]

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], v20 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_swmmac_f32_16x16x32_bf16 v[12:19], s[0:3], v[4:11], v20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], s[4:11], v20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], s20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[12:19], 1.0, v[4:11], v20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], 1.0, v20
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[12:19], v[0:3], v[4:11], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16
// GFX12: v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 ; encoding: [0x0c,0x40,0x52,0xcc,0x00,0x09,0x42,0x1c]

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 index_key:1
// GFX12: v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 index_key:1 ; encoding: [0x0c,0x48,0x52,0xcc,0x00,0x09,0x42,0x1c]

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[1,0,0]
// GFX12: v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[1,0,0] ; encoding: [0x0c,0x40,0x52,0xcc,0x00,0x09,0x42,0x3c]

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[0,1,0]
// GFX12: v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[0,1,0] ; encoding: [0x0c,0x40,0x52,0xcc,0x00,0x09,0x42,0x5c]

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[1,0,0]
// GFX12: v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[1,0,0] ; encoding: [0x0c,0x41,0x52,0xcc,0x00,0x09,0x42,0x1c]

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[0,1,0]
// GFX12: v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[0,1,0] ; encoding: [0x0c,0x42,0x52,0xcc,0x00,0x09,0x42,0x1c]

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_swmmac_f16_16x16x32_f16 v[12:15], s[0:3], v[4:11], v16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], s[4:11], v16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], s16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[12:15], 1.0, v[4:11], v16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], 1.0, v16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[12:15], v[0:3], v[4:11], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 ; encoding: [0x0c,0x40,0x53,0xcc,0x00,0x09,0x42,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 index_key:1
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 index_key:1 ; encoding: [0x0c,0x48,0x53,0xcc,0x00,0x09,0x42,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[1,0,0]
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[1,0,0] ; encoding: [0x0c,0x40,0x53,0xcc,0x00,0x09,0x42,0x3c]

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[0,1,0]
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[0,1,0] ; encoding: [0x0c,0x40,0x53,0xcc,0x00,0x09,0x42,0x5c]

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[1,0,0]
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[1,0,0] ; encoding: [0x0c,0x41,0x53,0xcc,0x00,0x09,0x42,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[0,1,0]
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[0,1,0] ; encoding: [0x0c,0x42,0x53,0xcc,0x00,0x09,0x42,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], v16 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_swmmac_bf16_16x16x32_bf16 v[12:15], s[0:3], v[4:11], v16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], s[4:11], v16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], s16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[12:15], 1.0, v[4:11], v16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], 1.0, v16
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[12:15], v[0:3], v[4:11], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14
// GFX12: v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 ; encoding: [0x06,0x40,0x54,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 clamp
// GFX12: v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 clamp ; encoding: [0x06,0xc0,0x54,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 index_key:1
// GFX12: v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 index_key:1 ; encoding: [0x06,0x48,0x54,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[1,0,0]
// GFX12: v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[1,0,0] ; encoding: [0x06,0x40,0x54,0xcc,0x00,0x05,0x3a,0x3c]

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,1,0]
// GFX12: v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,1,0] ; encoding: [0x06,0x40,0x54,0xcc,0x00,0x05,0x3a,0x5c]

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[6:13], s[0:1], v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], s[0:3], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], s14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[6:13], 1, v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], 1, v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[6:13], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11
// GFX12: v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 ; encoding: [0x03,0x40,0x55,0xcc,0x00,0x03,0x2e,0x1c]

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 clamp
// GFX12: v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 clamp ; encoding: [0x03,0xc0,0x55,0xcc,0x00,0x03,0x2e,0x1c]

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 index_key:1
// GFX12: v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 index_key:1 ; encoding: [0x03,0x48,0x55,0xcc,0x00,0x03,0x2e,0x1c]

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 neg_lo:[1,0,0]
// GFX12: v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 neg_lo:[1,0,0] ; encoding: [0x03,0x40,0x55,0xcc,0x00,0x03,0x2e,0x3c]

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 neg_lo:[0,1,0]
// GFX12: v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 neg_lo:[0,1,0] ; encoding: [0x03,0x40,0x55,0xcc,0x00,0x03,0x2e,0x5c]

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], v11 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[3:10], s0, v[1:2], v11
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, s[0:1], v11
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], s11
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[3:10], 1, v[1:2], v11
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, 1, v11
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[3:10], v0, v[1:2], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14
// GFX12: v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 ; encoding: [0x06,0x40,0x56,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 clamp
// GFX12: v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 clamp ; encoding: [0x06,0xc0,0x56,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 index_key:1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 neg_lo:[1,0,0]
// GFX12: v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 neg_lo:[1,0,0] ; encoding: [0x06,0x40,0x56,0xcc,0x00,0x05,0x3a,0x3c]

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,1,0]
// GFX12: v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,1,0] ; encoding: [0x06,0x40,0x56,0xcc,0x00,0x05,0x3a,0x5c]

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[6:13], s[0:1], v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], s[0:3], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], s14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[6:13], 1, v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], 1, v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[6:13], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14
// GFX12: v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 ; encoding: [0x06,0x40,0x57,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 index_key:1
// GFX12: v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 index_key:1 ; encoding: [0x06,0x48,0x57,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], s[0:1], v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], s[0:3], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], s14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], 1.0, v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], 1.0, v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[6:13], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14
// GFX12: v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 ; encoding: [0x06,0x40,0x58,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 index_key:1
// GFX12: v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 index_key:1 ; encoding: [0x06,0x48,0x58,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], s[0:1], v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], s[0:3], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], s14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], 1.0, v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], 1.0, v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[6:13], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14
// GFX12: v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 ; encoding: [0x06,0x40,0x59,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 index_key:1
// GFX12: v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 index_key:1 ; encoding: [0x06,0x48,0x59,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], s[0:1], v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], s[0:3], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], s14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], 1.0, v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], 1.0, v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[6:13], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14
// GFX12: v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 ; encoding: [0x06,0x40,0x5a,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 index_key:1
// GFX12: v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 index_key:1 ; encoding: [0x06,0x48,0x5a,0xcc,0x00,0x05,0x3a,0x1c]

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], v14 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], s[0:1], v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], s[0:3], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], s14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], 1.0, v[2:5], v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], 1.0, v14
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[6:13], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
