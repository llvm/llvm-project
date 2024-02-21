// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefix=GFX12 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: %s

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7]
// GFX12: v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] ; encoding: [0x04,0x40,0x40,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[1,0,0]
// GFX12: v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[1,0,0] ; encoding: [0x04,0x40,0x40,0xcc,0x00,0x05,0x12,0x3c]

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[0,1,0]
// GFX12: v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[0,1,0] ; encoding: [0x04,0x40,0x40,0xcc,0x00,0x05,0x12,0x5c]

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x40,0xcc,0x00,0x05,0x12,0x9c]

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[1,0,0]
// GFX12: v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[1,0,0] ; encoding: [0x04,0x41,0x40,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[0,1,0]
// GFX12: v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[0,1,0] ; encoding: [0x04,0x42,0x40,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[0,0,1] ; encoding: [0x04,0x44,0x40,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_f16 v[4:7], s[0:1], v[2:3], v[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], s[2:3], v[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], s[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[4:7], 1.0, v[2:3], v[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], 1.0, v[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x40,0xcc,0x00,0x05,0xca,0x1b]

v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], 1
// GFX12: v_wmma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x40,0xcc,0x00,0x05,0x06,0x1a]



v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7]
// GFX12: v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] ; encoding: [0x04,0x40,0x41,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[1,0,0]
// GFX12: v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[1,0,0] ; encoding: [0x04,0x40,0x41,0xcc,0x00,0x05,0x12,0x3c]

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[0,1,0]
// GFX12: v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[0,1,0] ; encoding: [0x04,0x40,0x41,0xcc,0x00,0x05,0x12,0x5c]

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x41,0xcc,0x00,0x05,0x12,0x9c]

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[1,0,0]
// GFX12: v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[1,0,0] ; encoding: [0x04,0x41,0x41,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[0,1,0]
// GFX12: v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[0,1,0] ; encoding: [0x04,0x42,0x41,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], v[4:7] neg_hi:[0,0,1] ; encoding: [0x04,0x44,0x41,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f32_16x16x16_bf16 v[4:7], s[0:1], v[2:3], v[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], s[2:3], v[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], s[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[4:7], 1.0, v[2:3], v[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], 1.0, v[4:7]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x41,0xcc,0x00,0x05,0xca,0x1b]

v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], 1
// GFX12: v_wmma_f32_16x16x16_bf16 v[4:7], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x41,0xcc,0x00,0x05,0x06,0x1a]



v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5]
// GFX12: v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] ; encoding: [0x04,0x40,0x42,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[1,0,0]
// GFX12: v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[1,0,0] ; encoding: [0x04,0x40,0x42,0xcc,0x00,0x05,0x12,0x3c]

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[0,1,0]
// GFX12: v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[0,1,0] ; encoding: [0x04,0x40,0x42,0xcc,0x00,0x05,0x12,0x5c]

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[0,0,1]
// GFX12: v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x42,0xcc,0x00,0x05,0x12,0x9c]

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[1,0,0]
// GFX12: v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[1,0,0] ; encoding: [0x04,0x41,0x42,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[0,1,0]
// GFX12: v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[0,1,0] ; encoding: [0x04,0x42,0x42,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[0,0,1]
// GFX12: v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[0,0,1] ; encoding: [0x04,0x44,0x42,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_f16_16x16x16_f16 v[4:5], s[0:1], v[2:3], v[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], s[2:3], v[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], s[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[4:5], 1.0, v[2:3], v[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], 1.0, v[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x42,0xcc,0x00,0x05,0xca,0x1b]

v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], 1
// GFX12: v_wmma_f16_16x16x16_f16 v[4:5], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x42,0xcc,0x00,0x05,0x06,0x1a]



v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] ; encoding: [0x04,0x40,0x43,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[1,0,0]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[1,0,0] ; encoding: [0x04,0x40,0x43,0xcc,0x00,0x05,0x12,0x3c]

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[0,1,0]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[0,1,0] ; encoding: [0x04,0x40,0x43,0xcc,0x00,0x05,0x12,0x5c]

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[0,0,1]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x43,0xcc,0x00,0x05,0x12,0x9c]

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[1,0,0]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[1,0,0] ; encoding: [0x04,0x41,0x43,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[0,1,0]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[0,1,0] ; encoding: [0x04,0x42,0x43,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[0,0,1]
// GFX12: v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], v[4:5] neg_hi:[0,0,1] ; encoding: [0x04,0x44,0x43,0xcc,0x00,0x05,0x12,0x1c]

v_wmma_bf16_16x16x16_bf16 v[4:5], s[0:1], v[2:3], v[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], s[2:3], v[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], s[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[4:5], 1.0, v[2:3], v[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], 1.0, v[4:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], 1.0
// GFX12: v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x43,0xcc,0x00,0x05,0xca,0x1b]

v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], 1
// GFX12: v_wmma_bf16_16x16x16_bf16 v[4:5], v[0:1], v[2:3], 1 ; encoding: [0x04,0x40,0x43,0xcc,0x00,0x05,0x06,0x1a]



v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5]
// GFX12: v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] ; encoding: [0x02,0x40,0x44,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] clamp
// GFX12: v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] clamp ; encoding: [0x02,0xc0,0x44,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0]
// GFX12: v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0] ; encoding: [0x02,0x40,0x44,0xcc,0x00,0x03,0x0a,0x3c]

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0]
// GFX12: v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0] ; encoding: [0x02,0x40,0x44,0xcc,0x00,0x03,0x0a,0x5c]

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu8 v[2:5], s0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[2:5], v0, s1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, s[0:3]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[2:5], 1, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[2:5], v0, 1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, 1
// GFX12: v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, 1 ; encoding: [0x02,0x40,0x44,0xcc,0x00,0x03,0x06,0x1a]

v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, 1.0
// GFX12: v_wmma_i32_16x16x16_iu8 v[2:5], v0, v1, 1.0 ; encoding: [0x02,0x40,0x44,0xcc,0x00,0x03,0xca,0x1b]



v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5]
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] clamp
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] clamp ; encoding: [0x02,0xc0,0x45,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0]
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0] ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0x0a,0x3c]

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0]
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0] ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0x0a,0x5c]

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x16_iu4 v[2:5], s0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:5], v0, s1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, s[0:3]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:5], 1, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:5], v0, 1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, 1
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, 1 ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0x06,0x1a]

v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, 1.0
// GFX12: v_wmma_i32_16x16x16_iu4 v[2:5], v0, v1, 1.0 ; encoding: [0x02,0x40,0x45,0xcc,0x00,0x03,0xca,0x1b]



v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5]
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] ; encoding: [0x02,0x40,0x46,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1] ; encoding: [0x02,0x40,0x46,0xcc,0x00,0x03,0x0a,0x9c]

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1] ; encoding: [0x02,0x44,0x46,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], s0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, s1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, s[0:3]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], 1.0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, 1.0, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, 1.0
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, 1.0 ; encoding: [0x02,0x40,0x46,0xcc,0x00,0x03,0xca,0x1b]

v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, 1
// GFX12: v_wmma_f32_16x16x16_fp8_fp8 v[2:5], v0, v1, 1 ; encoding: [0x02,0x40,0x46,0xcc,0x00,0x03,0x06,0x1a]



v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5]
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] ; encoding: [0x02,0x40,0x47,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1] ; encoding: [0x02,0x40,0x47,0xcc,0x00,0x03,0x0a,0x9c]

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1] ; encoding: [0x02,0x44,0x47,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], s0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, s1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, s[0:3]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], 1.0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, 1.0, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, 1.0
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, 1.0 ; encoding: [0x02,0x40,0x47,0xcc,0x00,0x03,0xca,0x1b]

v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, 1
// GFX12: v_wmma_f32_16x16x16_fp8_bf8 v[2:5], v0, v1, 1 ; encoding: [0x02,0x40,0x47,0xcc,0x00,0x03,0x06,0x1a]



v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5]
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] ; encoding: [0x02,0x40,0x48,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1] ; encoding: [0x02,0x40,0x48,0xcc,0x00,0x03,0x0a,0x9c]

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1] ; encoding: [0x02,0x44,0x48,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], s0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, s1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, s[0:3]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], 1.0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, 1.0, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, 1.0
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, 1.0 ; encoding: [0x02,0x40,0x48,0xcc,0x00,0x03,0xca,0x1b]

v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, 1
// GFX12: v_wmma_f32_16x16x16_bf8_fp8 v[2:5], v0, v1, 1 ; encoding: [0x02,0x40,0x48,0xcc,0x00,0x03,0x06,0x1a]



v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5]
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] ; encoding: [0x02,0x40,0x49,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1] ; encoding: [0x02,0x40,0x49,0xcc,0x00,0x03,0x0a,0x9c]

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1]
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1] ; encoding: [0x02,0x44,0x49,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], s0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, s1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, s[0:3]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], 1.0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, 1.0, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, 1.0
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, 1.0 ; encoding: [0x02,0x40,0x49,0xcc,0x00,0x03,0xca,0x1b]

v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, 1
// GFX12: v_wmma_f32_16x16x16_bf8_bf8 v[2:5], v0, v1, 1 ; encoding: [0x02,0x40,0x49,0xcc,0x00,0x03,0x06,0x1a]



v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5]
// GFX12: v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] ; encoding: [0x02,0x40,0x4a,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] clamp
// GFX12: v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] clamp ; encoding: [0x02,0xc0,0x4a,0xcc,0x00,0x03,0x0a,0x1c]

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0]
// GFX12: v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[1,0,0] ; encoding: [0x02,0x40,0x4a,0xcc,0x00,0x03,0x0a,0x3c]

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0]
// GFX12: v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[0,1,0] ; encoding: [0x02,0x40,0x4a,0xcc,0x00,0x03,0x0a,0x5c]

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, v[2:5] neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_wmma_i32_16x16x32_iu4 v[2:5], s0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[2:5], v0, s1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, s[0:3]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[2:5], 1.0, v1, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[2:5], v0, 1.0, v[2:5]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, 1.0
// GFX12: v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, 1.0 ; encoding: [0x02,0x40,0x4a,0xcc,0x00,0x03,0xca,0x1b]

v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, 1
// GFX12: v_wmma_i32_16x16x32_iu4 v[2:5], v0, v1, 1 ; encoding: [0x02,0x40,0x4a,0xcc,0x00,0x03,0x06,0x1a]



v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10
// GFX12: v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 ; encoding: [0x06,0x40,0x50,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 index_key:1
// GFX12: v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 index_key:1 ; encoding: [0x06,0x48,0x50,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 index_key:2
// GFX12: v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 index_key:2 ; encoding: [0x06,0x50,0x50,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 index_key:3
// GFX12: v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 index_key:3 ; encoding: [0x06,0x58,0x50,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[1,0,0]
// GFX12: v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[1,0,0] ; encoding: [0x06,0x40,0x50,0xcc,0x00,0x05,0x2a,0x3c]

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[0,1,0]
// GFX12: v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[0,1,0] ; encoding: [0x06,0x40,0x50,0xcc,0x00,0x05,0x2a,0x5c]

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[1,0,0]
// GFX12: v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[1,0,0] ; encoding: [0x06,0x41,0x50,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[0,1,0]
// GFX12: v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[0,1,0] ; encoding: [0x06,0x42,0x50,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_swmmac_f32_16x16x32_f16 v[6:9], s[0:1], v[2:5], v10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], s[0:3], v10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], s10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[6:9], 1.0, v[2:5], v10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], 1.0, v10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_f16 v[6:9], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10
// GFX12: v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 ; encoding: [0x06,0x40,0x51,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 index_key:1
// GFX12: v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 index_key:1 ; encoding: [0x06,0x48,0x51,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 index_key:2
// GFX12: v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 index_key:2 ; encoding: [0x06,0x50,0x51,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 index_key:3
// GFX12: v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 index_key:3 ; encoding: [0x06,0x58,0x51,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[1,0,0]
// GFX12: v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[1,0,0] ; encoding: [0x06,0x40,0x51,0xcc,0x00,0x05,0x2a,0x3c]

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[0,1,0]
// GFX12: v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[0,1,0] ; encoding: [0x06,0x40,0x51,0xcc,0x00,0x05,0x2a,0x5c]

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[1,0,0]
// GFX12: v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[1,0,0] ; encoding: [0x06,0x41,0x51,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[0,1,0]
// GFX12: v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[0,1,0] ; encoding: [0x06,0x42,0x51,0xcc,0x00,0x05,0x2a,0x1c]

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], v10 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_swmmac_f32_16x16x32_bf16 v[6:9], s[0:1], v[2:5], v10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], s[0:3], v10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], s10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[6:9], 1.0, v[2:5], v10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], 1.0, v10
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf16 v[6:9], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8
// GFX12: v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 ; encoding: [0x06,0x40,0x52,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 index_key:1
// GFX12: v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 index_key:1 ; encoding: [0x06,0x48,0x52,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 index_key:2
// GFX12: v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 index_key:2 ; encoding: [0x06,0x50,0x52,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 index_key:3
// GFX12: v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 index_key:3 ; encoding: [0x06,0x58,0x52,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[1,0,0]
// GFX12: v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[1,0,0] ; encoding: [0x06,0x40,0x52,0xcc,0x00,0x05,0x22,0x3c]

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[0,1,0]
// GFX12: v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[0,1,0] ; encoding: [0x06,0x40,0x52,0xcc,0x00,0x05,0x22,0x5c]

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[1,0,0]
// GFX12: v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[1,0,0] ; encoding: [0x06,0x41,0x52,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[0,1,0]
// GFX12: v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[0,1,0] ; encoding: [0x06,0x42,0x52,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_swmmac_f16_16x16x32_f16 v[6:7], s[0:1], v[2:5], v8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], s[0:3], v8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], s8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[6:7], 1.0, v[2:5], v8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], 1.0, v8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f16_16x16x32_f16 v[6:7], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 ; encoding: [0x06,0x40,0x53,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 index_key:1
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 index_key:1 ; encoding: [0x06,0x48,0x53,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 index_key:2
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 index_key:2 ; encoding: [0x06,0x50,0x53,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 index_key:3
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 index_key:3 ; encoding: [0x06,0x58,0x53,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[1,0,0]
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[1,0,0] ; encoding: [0x06,0x40,0x53,0xcc,0x00,0x05,0x22,0x3c]

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[0,1,0]
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[0,1,0] ; encoding: [0x06,0x40,0x53,0xcc,0x00,0x05,0x22,0x5c]

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[1,0,0]
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[1,0,0] ; encoding: [0x06,0x41,0x53,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[0,1,0]
// GFX12: v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[0,1,0] ; encoding: [0x06,0x42,0x53,0xcc,0x00,0x05,0x22,0x1c]

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], v8 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_hi operand

v_swmmac_bf16_16x16x32_bf16 v[6:7], s[0:1], v[2:5], v8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], s[0:3], v8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], s8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[6:7], 1.0, v[2:5], v8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], 1.0, v8
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_bf16_16x16x32_bf16 v[6:7], v[0:1], v[2:5], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7
// GFX12: v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 ; encoding: [0x03,0x40,0x54,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 clamp
// GFX12: v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 clamp ; encoding: [0x03,0xc0,0x54,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 index_key:1
// GFX12: v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 index_key:1 ; encoding: [0x03,0x48,0x54,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 index_key:2
// GFX12: v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 index_key:2 ; encoding: [0x03,0x50,0x54,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 index_key:3
// GFX12: v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 index_key:3 ; encoding: [0x03,0x58,0x54,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 neg_lo:[1,0,0]
// GFX12: v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 neg_lo:[1,0,0] ; encoding: [0x03,0x40,0x54,0xcc,0x00,0x03,0x1e,0x3c]

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 neg_lo:[0,1,0]
// GFX12: v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 neg_lo:[0,1,0] ; encoding: [0x03,0x40,0x54,0xcc,0x00,0x03,0x1e,0x5c]

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], v7 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu8 v[3:6], s0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, s[0:1], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], s7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[3:6], 1, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, 1, v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu8 v[3:6], v0, v[1:2], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6
// GFX12: v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 ; encoding: [0x02,0x40,0x55,0xcc,0x00,0x03,0x1a,0x1c]

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 clamp
// GFX12: v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 clamp ; encoding: [0x02,0xc0,0x55,0xcc,0x00,0x03,0x1a,0x1c]

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 index_key:1
// GFX12: v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 index_key:1 ; encoding: [0x02,0x48,0x55,0xcc,0x00,0x03,0x1a,0x1c]

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 neg_lo:[1,0,0]
// GFX12: v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 neg_lo:[1,0,0] ; encoding: [0x02,0x40,0x55,0xcc,0x00,0x03,0x1a,0x3c]

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 neg_lo:[0,1,0]
// GFX12: v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 neg_lo:[0,1,0] ; encoding: [0x02,0x40,0x55,0xcc,0x00,0x03,0x1a,0x5c]

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, v6 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x32_iu4 v[2:5], s0, v1, v6
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, s1, v6
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, s6
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[2:5], 1, v1, v6
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, 1, v6
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x32_iu4 v[2:5], v0, v1, 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7
// GFX12: v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 ; encoding: [0x03,0x40,0x56,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 clamp
// GFX12: v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 clamp ; encoding: [0x03,0xc0,0x56,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 index_key:1
// GFX12: v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 index_key:1 ; encoding: [0x03,0x48,0x56,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 index_key:2
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 index_key:3
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: out of range index_key

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 neg_lo:[1,0,0]
// GFX12: v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 neg_lo:[1,0,0] ; encoding: [0x03,0x40,0x56,0xcc,0x00,0x03,0x1e,0x3c]

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 neg_lo:[0,1,0]
// GFX12: v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 neg_lo:[0,1,0] ; encoding: [0x03,0x40,0x56,0xcc,0x00,0x03,0x1e,0x5c]

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid neg_lo operand

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], v7 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_i32_16x16x64_iu4 v[3:6], s0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, s[0:1], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], s7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[3:6], 1, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, 1, v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_i32_16x16x64_iu4 v[3:6], v0, v[1:2], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7
// GFX12: v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 ; encoding: [0x03,0x40,0x57,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 index_key:1
// GFX12: v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 index_key:1 ; encoding: [0x03,0x48,0x57,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 index_key:2
// GFX12: v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 index_key:2 ; encoding: [0x03,0x50,0x57,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 index_key:3
// GFX12: v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 index_key:3 ; encoding: [0x03,0x58,0x57,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], v7 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], s0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, s[0:1], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], s7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], 1.0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, 1.0, v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_fp8 v[3:6], v0, v[1:2], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7
// GFX12: v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 ; encoding: [0x03,0x40,0x58,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 index_key:1
// GFX12: v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 index_key:1 ; encoding: [0x03,0x48,0x58,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 index_key:2
// GFX12: v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 index_key:2 ; encoding: [0x03,0x50,0x58,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 index_key:3
// GFX12: v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 index_key:3 ; encoding: [0x03,0x58,0x58,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], v7 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], s0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, s[0:1], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], s7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], 1.0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, 1.0, v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_fp8_bf8 v[3:6], v0, v[1:2], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7
// GFX12: v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 ; encoding: [0x03,0x40,0x59,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 index_key:1
// GFX12: v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 index_key:1 ; encoding: [0x03,0x48,0x59,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 index_key:2
// GFX12: v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 index_key:2 ; encoding: [0x03,0x50,0x59,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 index_key:3
// GFX12: v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 index_key:3 ; encoding: [0x03,0x58,0x59,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], v7 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], s0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, s[0:1], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], s7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], 1.0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, 1.0, v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_fp8 v[3:6], v0, v[1:2], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction



v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7
// GFX12: v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 ; encoding: [0x03,0x40,0x5a,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 clamp
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 op_sel:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 op_sel:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 op_sel:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 op_sel_hi:[0,1,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 op_sel_hi:[1,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 index_key:1
// GFX12: v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 index_key:1 ; encoding: [0x03,0x48,0x5a,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 index_key:2
// GFX12: v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 index_key:2 ; encoding: [0x03,0x50,0x5a,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 index_key:3
// GFX12: v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 index_key:3 ; encoding: [0x03,0x58,0x5a,0xcc,0x00,0x03,0x1e,0x1c]

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 neg_lo:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 neg_lo:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 neg_lo:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 neg_hi:[1,0,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 neg_hi:[0,1,0]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], v7 neg_hi:[0,0,1]
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], s0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, s[0:1], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], s7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], 1.0, v[1:2], v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, 1.0, v7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], 1.0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_swmmac_f32_16x16x32_bf8_bf8 v[3:6], v0, v[1:2], 1
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
