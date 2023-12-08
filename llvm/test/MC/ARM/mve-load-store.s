# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -show-encoding < %s 2>%t
# RUN:     FileCheck --check-prefix=ERROR-NOMVE < %t %s

# CHECK: vldrb.u8 q0, [r0] @ encoding: [0x90,0xed,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q0, [r0]

# CHECK: vldrb.u8 q1, [r0] @ encoding: [0x90,0xed,0x00,0x3e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q1, [r0]

# CHECK: vldrb.u8 q0, [r11] @ encoding: [0x9b,0xed,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q0, [r11]

# CHECK: vldrb.u8 q3, [r11] @ encoding: [0x9b,0xed,0x00,0x7e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q3, [r11]

# CHECK: vldrb.u8 q0, [r4, #56] @ encoding: [0x94,0xed,0x38,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q0, [r4, #56]

# CHECK: vldrb.u8 q4, [r4, #56] @ encoding: [0x94,0xed,0x38,0x9e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q4, [r4, #56]

# CHECK: vldrb.u8 q0, [r8, #56] @ encoding: [0x98,0xed,0x38,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q0, [r8, #56]

# CHECK: vldrb.u8 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x38,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q5, [r4, #56]!

# CHECK: vldrb.u8 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x38,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q5, [r4, #56]!

# CHECK: vldrb.u8 q5, [r4], #-25 @ encoding: [0x34,0xec,0x19,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q5, [r4], #-25

# CHECK: vldrb.u8 q5, [r10], #-25 @ encoding: [0x3a,0xec,0x19,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q5, [r10], #-25

# CHECK: vldrb.u8 q5, [sp, #-25] @ encoding: [0x1d,0xed,0x19,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q5, [sp, #-25]

# CHECK: vldrb.u8 q5, [sp, #-127] @ encoding: [0x1d,0xed,0x7f,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q5, [sp, #-127]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q0, [r0, #128]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q0, [r0, #-128]!

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q0, [r0], #128

# CHECK: vstrb.8 q0, [r0] @ encoding: [0x80,0xed,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q0, [r0]

# CHECK: vstrb.8 q1, [r0] @ encoding: [0x80,0xed,0x00,0x3e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q1, [r0]

# CHECK: vstrb.8 q0, [r11] @ encoding: [0x8b,0xed,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q0, [r11]

# CHECK: vstrb.8 q3, [r11] @ encoding: [0x8b,0xed,0x00,0x7e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q3, [r11]

# CHECK: vstrb.8 q0, [r4, #56] @ encoding: [0x84,0xed,0x38,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q0, [r4, #56]

# CHECK: vstrb.8 q4, [r4, #56] @ encoding: [0x84,0xed,0x38,0x9e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q4, [r4, #56]

# CHECK: vstrb.8 q0, [r8, #56] @ encoding: [0x88,0xed,0x38,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q0, [r8, #56]

# CHECK: vstrb.8 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x38,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q5, [r4, #56]!

# CHECK: vstrb.8 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x38,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q5, [r4, #56]!

# CHECK: vstrb.8 q5, [r4], #-25 @ encoding: [0x24,0xec,0x19,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q5, [r4], #-25

# CHECK: vstrb.8 q5, [r10], #-25 @ encoding: [0x2a,0xec,0x19,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q5, [r10], #-25

# CHECK: vstrb.8 q5, [sp, #-25] @ encoding: [0x0d,0xed,0x19,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q5, [sp, #-25]

# CHECK: vstrb.8 q5, [sp, #127] @ encoding: [0x8d,0xed,0x7f,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q5, [sp, #127]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.u8 q0, [r0, #128]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.u8 q0, [r0, #-128]!

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.u8 q0, [r0], #128

# CHECK: vldrb.u16 q0, [r0] @ encoding: [0x90,0xfd,0x80,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q0, [r0]

# CHECK: vldrb.u16 q1, [r0] @ encoding: [0x90,0xfd,0x80,0x2e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q1, [r0]

# CHECK: vldrb.u16 q0, [r7] @ encoding: [0x97,0xfd,0x80,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q0, [r7]

# CHECK: vldrb.u16 q3, [r7] @ encoding: [0x97,0xfd,0x80,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q3, [r7]

# CHECK: vldrb.u16 q0, [r4, #56] @ encoding: [0x94,0xfd,0xb8,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q0, [r4, #56]

# CHECK: vldrb.u16 q4, [r4, #56] @ encoding: [0x94,0xfd,0xb8,0x8e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q4, [r4, #56]

# CHECK: vldrb.u16 q0, [r2, #56] @ encoding: [0x92,0xfd,0xb8,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q0, [r2, #56]

# CHECK: vldrb.u16 q5, [r4, #56]! @ encoding: [0xb4,0xfd,0xb8,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q5, [r4, #56]!

# CHECK: vldrb.u16 q5, [r4, #56]! @ encoding: [0xb4,0xfd,0xb8,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q5, [r4, #56]!

# CHECK: vldrb.u16 q5, [r4], #-1 @ encoding: [0x34,0xfc,0x81,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q5, [r4], #-1

# CHECK: vldrb.u16 q5, [r3], #-25 @ encoding: [0x33,0xfc,0x99,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q5, [r3], #-25

# CHECK: vldrb.u16 q5, [r6, #-25] @ encoding: [0x16,0xfd,0x99,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q5, [r6, #-25]

# CHECK: vldrb.u16 q5, [r6, #-64] @ encoding: [0x16,0xfd,0xc0,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q5, [r6, #-64]

# CHECK: vldrb.s16 q0, [r0] @ encoding: [0x90,0xed,0x80,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q0, [r0]

# CHECK: vldrb.s16 q1, [r0] @ encoding: [0x90,0xed,0x80,0x2e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q1, [r0]

# CHECK: vldrb.s16 q0, [r7] @ encoding: [0x97,0xed,0x80,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q0, [r7]

# CHECK: vldrb.s16 q3, [r7] @ encoding: [0x97,0xed,0x80,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q3, [r7]

# CHECK: vldrb.s16 q0, [r4, #56] @ encoding: [0x94,0xed,0xb8,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q0, [r4, #56]

# CHECK: vldrb.s16 q4, [r4, #56] @ encoding: [0x94,0xed,0xb8,0x8e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q4, [r4, #56]

# CHECK: vldrb.s16 q0, [r2, #56] @ encoding: [0x92,0xed,0xb8,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q0, [r2, #56]

# CHECK: vldrb.s16 q5, [r4, #56]! @ encoding: [0xb4,0xed,0xb8,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q5, [r4, #56]!

# CHECK: vldrb.s16 q5, [r4, #56]! @ encoding: [0xb4,0xed,0xb8,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q5, [r4, #56]!

# CHECK: vldrb.s16 q5, [r4], #-25 @ encoding: [0x34,0xec,0x99,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q5, [r4], #-25

# CHECK: vldrb.s16 q5, [r3], #-25 @ encoding: [0x33,0xec,0x99,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q5, [r3], #-25

# CHECK: vldrb.s16 q5, [r6, #-25] @ encoding: [0x16,0xed,0x99,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q5, [r6, #-25]

# CHECK: vldrb.s16 q5, [r6, #-64] @ encoding: [0x16,0xed,0xc0,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q5, [r6, #-64]

# CHECK: vstrb.16 q0, [r0] @ encoding: [0x80,0xed,0x80,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q0, [r0]

# CHECK: vstrb.16 q1, [r0] @ encoding: [0x80,0xed,0x80,0x2e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q1, [r0]

# CHECK: vstrb.16 q0, [r7] @ encoding: [0x87,0xed,0x80,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q0, [r7]

# CHECK: vstrb.16 q3, [r7] @ encoding: [0x87,0xed,0x80,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q3, [r7]

# CHECK: vstrb.16 q0, [r4, #56] @ encoding: [0x84,0xed,0xb8,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q0, [r4, #56]

# CHECK: vstrb.16 q4, [r4, #56] @ encoding: [0x84,0xed,0xb8,0x8e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q4, [r4, #56]

# CHECK: vstrb.16 q0, [r5, #56] @ encoding: [0x85,0xed,0xb8,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q0, [r5, #56]

# CHECK: vstrb.16 q5, [r4, #56]! @ encoding: [0xa4,0xed,0xb8,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q5, [r4, #56]!

# CHECK: vstrb.16 q5, [r4, #56]! @ encoding: [0xa4,0xed,0xb8,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q5, [r4, #56]!

# CHECK: vstrb.16 q5, [r4], #-25 @ encoding: [0x24,0xec,0x99,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q5, [r4], #-25

# CHECK: vstrb.16 q5, [r3], #-25 @ encoding: [0x23,0xec,0x99,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q5, [r3], #-25

# CHECK: vstrb.16 q5, [r2, #-25] @ encoding: [0x02,0xed,0x99,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q5, [r2, #-25]

# CHECK: vstrb.16 q5, [r2, #-64] @ encoding: [0x02,0xed,0xc0,0xae]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q5, [r2, #-64]

# CHECK: vldrb.u32 q0, [r0] @ encoding: [0x90,0xfd,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q0, [r0]

# CHECK: vldrb.u32 q1, [r0] @ encoding: [0x90,0xfd,0x00,0x2f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q1, [r0]

# CHECK: vldrb.u32 q0, [r7] @ encoding: [0x97,0xfd,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q0, [r7]

# CHECK: vldrb.u32 q3, [r7] @ encoding: [0x97,0xfd,0x00,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q3, [r7]

# CHECK: vldrb.u32 q0, [r4, #56] @ encoding: [0x94,0xfd,0x38,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q0, [r4, #56]

# CHECK: vldrb.u32 q4, [r4, #56] @ encoding: [0x94,0xfd,0x38,0x8f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q4, [r4, #56]

# CHECK: vldrb.u32 q0, [r2, #56] @ encoding: [0x92,0xfd,0x38,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q0, [r2, #56]

# CHECK: vldrb.u32 q5, [r4, #56]! @ encoding: [0xb4,0xfd,0x38,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q5, [r4, #56]!

# CHECK: vldrb.u32 q5, [r4, #56]! @ encoding: [0xb4,0xfd,0x38,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q5, [r4, #56]!

# CHECK: vldrb.u32 q5, [r4], #-25 @ encoding: [0x34,0xfc,0x19,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q5, [r4], #-25

# CHECK: vldrb.u32 q5, [r3], #-25 @ encoding: [0x33,0xfc,0x19,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q5, [r3], #-25

# CHECK: vldrb.u32 q5, [r6, #-25] @ encoding: [0x16,0xfd,0x19,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q5, [r6, #-25]

# CHECK: vldrb.u32 q5, [r6, #-64] @ encoding: [0x16,0xfd,0x40,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q5, [r6, #-64]

# CHECK: vldrb.s32 q0, [r0] @ encoding: [0x90,0xed,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q0, [r0]

# CHECK: vldrb.s32 q1, [r0] @ encoding: [0x90,0xed,0x00,0x2f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q1, [r0]

# CHECK: vldrb.s32 q0, [r7] @ encoding: [0x97,0xed,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q0, [r7]

# CHECK: vldrb.s32 q3, [r7] @ encoding: [0x97,0xed,0x00,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q3, [r7]

# CHECK: vldrb.s32 q0, [r4, #56] @ encoding: [0x94,0xed,0x38,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q0, [r4, #56]

# CHECK: vldrb.s32 q4, [r4, #56] @ encoding: [0x94,0xed,0x38,0x8f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q4, [r4, #56]

# CHECK: vldrb.s32 q0, [r2, #56] @ encoding: [0x92,0xed,0x38,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q0, [r2, #56]

# CHECK: vldrb.s32 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x38,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q5, [r4, #56]!

# CHECK: vldrb.s32 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x38,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q5, [r4, #56]!

# CHECK: vldrb.s32 q5, [r4], #-25 @ encoding: [0x34,0xec,0x19,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q5, [r4], #-25

# CHECK: vldrb.s32 q5, [r3], #-25 @ encoding: [0x33,0xec,0x19,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q5, [r3], #-25

# CHECK: vldrb.s32 q5, [r6, #-25] @ encoding: [0x16,0xed,0x19,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q5, [r6, #-25]

# CHECK: vldrb.s32 q5, [r6, #-64] @ encoding: [0x16,0xed,0x40,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q5, [r6, #-64]

# CHECK: vstrb.32 q0, [r0] @ encoding: [0x80,0xed,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q0, [r0]

# CHECK: vstrb.32 q1, [r0] @ encoding: [0x80,0xed,0x00,0x2f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q1, [r0]

# CHECK: vstrb.32 q0, [r7] @ encoding: [0x87,0xed,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q0, [r7]

# CHECK: vstrb.32 q3, [r7] @ encoding: [0x87,0xed,0x00,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q3, [r7]

# CHECK: vstrb.32 q0, [r4, #56] @ encoding: [0x84,0xed,0x38,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q0, [r4, #56]

# CHECK: vstrb.32 q4, [r4, #56] @ encoding: [0x84,0xed,0x38,0x8f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q4, [r4, #56]

# CHECK: vstrb.32 q0, [r5, #56] @ encoding: [0x85,0xed,0x38,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q0, [r5, #56]

# CHECK: vstrb.32 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x38,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q5, [r4, #56]!

# CHECK: vstrb.32 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x38,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q5, [r4, #56]!

# CHECK: vstrb.32 q5, [r4], #-25 @ encoding: [0x24,0xec,0x19,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q5, [r4], #-25

# CHECK: vstrb.32 q5, [r3], #-25 @ encoding: [0x23,0xec,0x19,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q5, [r3], #-25

# CHECK: vstrb.32 q5, [r2, #-25] @ encoding: [0x02,0xed,0x19,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q5, [r2, #-25]

# CHECK: vstrb.32 q5, [r2, #-64] @ encoding: [0x02,0xed,0x40,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q5, [r2, #-64]

# CHECK: vldrh.u16 q0, [r0] @ encoding: [0x90,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q0, [r0]

# CHECK: vldrh.u16 q1, [r0] @ encoding: [0x90,0xed,0x80,0x3e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q1, [r0]

# CHECK: vldrh.u16 q0, [r11] @ encoding: [0x9b,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q0, [r11]

# CHECK: vldrh.u16 q3, [r11] @ encoding: [0x9b,0xed,0x80,0x7e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q3, [r11]

# CHECK: vldrh.u16 q0, [r4, #56] @ encoding: [0x94,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q0, [r4, #56]

# CHECK: vldrh.u16 q4, [r4, #56] @ encoding: [0x94,0xed,0x9c,0x9e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q4, [r4, #56]

# CHECK: vldrh.u16 q0, [r8, #56] @ encoding: [0x98,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q0, [r8, #56]

# CHECK: vldrh.u16 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q5, [r4, #56]!

# CHECK: vldrh.u16 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q5, [r4, #56]!

# CHECK: vldrh.u16 q5, [r4], #-26 @ encoding: [0x34,0xec,0x8d,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q5, [r4], #-26

# CHECK: vldrh.u16 q5, [r10], #-26 @ encoding: [0x3a,0xec,0x8d,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q5, [r10], #-26

# CHECK: vldrh.u16 q5, [sp, #-26] @ encoding: [0x1d,0xed,0x8d,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q5, [sp, #-26]

# CHECK: vldrh.u16 q5, [sp, #-64] @ encoding: [0x1d,0xed,0xa0,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q5, [sp, #-64]

# CHECK: vldrh.u16 q5, [sp, #-254] @ encoding: [0x1d,0xed,0xff,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q5, [sp, #-254]

# CHECK: vldrh.u16 q5, [r10], #254 @ encoding: [0xba,0xec,0xff,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q5, [r10], #254

# CHECK: vstrh.16 q0, [r0] @ encoding: [0x80,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q0, [r0]

# CHECK: vstrh.16 q1, [r0] @ encoding: [0x80,0xed,0x80,0x3e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q1, [r0]

# CHECK: vstrh.16 q0, [r11] @ encoding: [0x8b,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q0, [r11]

# CHECK: vstrh.16 q3, [r11] @ encoding: [0x8b,0xed,0x80,0x7e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q3, [r11]

# CHECK: vstrh.16 q0, [r4, #56] @ encoding: [0x84,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q0, [r4, #56]

# CHECK: vstrh.16 q4, [r4, #56] @ encoding: [0x84,0xed,0x9c,0x9e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q4, [r4, #56]

# CHECK: vstrh.16 q0, [r8, #56] @ encoding: [0x88,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q0, [r8, #56]

# CHECK: vstrh.16 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q5, [r4, #56]!

# CHECK: vstrh.16 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q5, [r4, #56]!

# CHECK: vstrh.16 q5, [r4], #-26 @ encoding: [0x24,0xec,0x8d,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q5, [r4], #-26

# CHECK: vstrh.16 q5, [r10], #-26 @ encoding: [0x2a,0xec,0x8d,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q5, [r10], #-26

# CHECK: vstrh.16 q5, [sp, #-26] @ encoding: [0x0d,0xed,0x8d,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q5, [sp, #-26]

# CHECK: vstrh.16 q5, [sp, #-64] @ encoding: [0x0d,0xed,0xa0,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q5, [sp, #-64]

# CHECK: vstrh.16 q5, [sp, #-254] @ encoding: [0x0d,0xed,0xff,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q5, [sp, #-254]

# CHECK: vstrh.16 q5, [r10], #254 @ encoding: [0xaa,0xec,0xff,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q5, [r10], #254

# CHECK: vldrh.u32 q0, [r0] @ encoding: [0x98,0xfd,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q0, [r0]

# CHECK: vldrh.u32 q1, [r0] @ encoding: [0x98,0xfd,0x00,0x2f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q1, [r0]

# CHECK: vldrh.u32 q0, [r7] @ encoding: [0x9f,0xfd,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q0, [r7]

# CHECK: vldrh.u32 q3, [r7] @ encoding: [0x9f,0xfd,0x00,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q3, [r7]

# CHECK: vldrh.u32 q0, [r4, #56] @ encoding: [0x9c,0xfd,0x1c,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q0, [r4, #56]

# CHECK: vldrh.u32 q4, [r4, #56] @ encoding: [0x9c,0xfd,0x1c,0x8f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q4, [r4, #56]

# CHECK: vldrh.u32 q0, [r2, #56] @ encoding: [0x9a,0xfd,0x1c,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q0, [r2, #56]

# CHECK: vldrh.u32 q5, [r4, #56]! @ encoding: [0xbc,0xfd,0x1c,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q5, [r4, #56]!

# CHECK: vldrh.u32 q5, [r4, #56]! @ encoding: [0xbc,0xfd,0x1c,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q5, [r4, #56]!

# CHECK: vldrh.u32 q5, [r4], #-26 @ encoding: [0x3c,0xfc,0x0d,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q5, [r4], #-26

# CHECK: vldrh.u32 q5, [r3], #-26 @ encoding: [0x3b,0xfc,0x0d,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q5, [r3], #-26

# CHECK: vldrh.u32 q5, [r6, #-26] @ encoding: [0x1e,0xfd,0x0d,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q5, [r6, #-26]

# CHECK: vldrh.u32 q5, [r6, #-64] @ encoding: [0x1e,0xfd,0x20,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q5, [r6, #-64]

# CHECK: vldrh.u32 q5, [r6, #-254] @ encoding: [0x1e,0xfd,0x7f,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q5, [r6, #-254]

# CHECK: vldrh.u32 q5, [r4, #254]! @ encoding: [0xbc,0xfd,0x7f,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q5, [r4, #254]!

# CHECK: vldrh.s32 q0, [r0] @ encoding: [0x98,0xed,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q0, [r0]

# CHECK: vldrh.s32 q1, [r0] @ encoding: [0x98,0xed,0x00,0x2f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q1, [r0]

# CHECK: vldrh.s32 q0, [r7] @ encoding: [0x9f,0xed,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q0, [r7]

# CHECK: vldrh.s32 q3, [r7] @ encoding: [0x9f,0xed,0x00,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q3, [r7]

# CHECK: vldrh.s32 q0, [r4, #56] @ encoding: [0x9c,0xed,0x1c,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q0, [r4, #56]

# CHECK: vldrh.s32 q4, [r4, #56] @ encoding: [0x9c,0xed,0x1c,0x8f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q4, [r4, #56]

# CHECK: vldrh.s32 q0, [r2, #56] @ encoding: [0x9a,0xed,0x1c,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q0, [r2, #56]

# CHECK: vldrh.s32 q5, [r4, #56]! @ encoding: [0xbc,0xed,0x1c,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q5, [r4, #56]!

# CHECK: vldrh.s32 q5, [r4, #56]! @ encoding: [0xbc,0xed,0x1c,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q5, [r4, #56]!

# CHECK: vldrh.s32 q5, [r4], #-26 @ encoding: [0x3c,0xec,0x0d,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q5, [r4], #-26

# CHECK: vldrh.s32 q5, [r3], #-26 @ encoding: [0x3b,0xec,0x0d,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q5, [r3], #-26

# CHECK: vldrh.s32 q5, [r6, #-26] @ encoding: [0x1e,0xed,0x0d,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q5, [r6, #-26]

# CHECK: vldrh.s32 q5, [r6, #-64] @ encoding: [0x1e,0xed,0x20,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q5, [r6, #-64]

# CHECK: vldrh.s32 q5, [r6, #-254] @ encoding: [0x1e,0xed,0x7f,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q5, [r6, #-254]

# CHECK: vldrh.s32 q5, [r4, #254]! @ encoding: [0xbc,0xed,0x7f,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q5, [r4, #254]!

# CHECK: vstrh.32 q0, [r0] @ encoding: [0x88,0xed,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q0, [r0]

# CHECK: vstrh.32 q1, [r0] @ encoding: [0x88,0xed,0x00,0x2f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q1, [r0]

# CHECK: vstrh.32 q0, [r7] @ encoding: [0x8f,0xed,0x00,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q0, [r7]

# CHECK: vstrh.32 q3, [r7] @ encoding: [0x8f,0xed,0x00,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q3, [r7]

# CHECK: vstrh.32 q0, [r4, #56] @ encoding: [0x8c,0xed,0x1c,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q0, [r4, #56]

# CHECK: vstrh.32 q4, [r4, #56] @ encoding: [0x8c,0xed,0x1c,0x8f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q4, [r4, #56]

# CHECK: vstrh.32 q0, [r5, #56] @ encoding: [0x8d,0xed,0x1c,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q0, [r5, #56]

# CHECK: vstrh.32 q5, [r4, #56]! @ encoding: [0xac,0xed,0x1c,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q5, [r4, #56]!

# CHECK: vstrh.32 q5, [r4, #56]! @ encoding: [0xac,0xed,0x1c,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q5, [r4, #56]!

# CHECK: vstrh.32 q5, [r4], #-26 @ encoding: [0x2c,0xec,0x0d,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q5, [r4], #-26

# CHECK: vstrh.32 q5, [r3], #-26 @ encoding: [0x2b,0xec,0x0d,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q5, [r3], #-26

# CHECK: vstrh.32 q5, [r2, #-26] @ encoding: [0x0a,0xed,0x0d,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q5, [r2, #-26]

# CHECK: vstrh.32 q5, [r2, #-64] @ encoding: [0x0a,0xed,0x20,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q5, [r2, #-64]

# CHECK: vstrh.32 q5, [r2, #-254] @ encoding: [0x0a,0xed,0x7f,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q5, [r2, #-254]

# CHECK: vstrh.32 q5, [r4, #254]! @ encoding: [0xac,0xed,0x7f,0xaf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q5, [r4, #254]!

# CHECK: vldrw.u32 q0, [r0] @ encoding: [0x90,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [r0]

# CHECK: vldrw.u32 q1, [r0] @ encoding: [0x90,0xed,0x00,0x3f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q1, [r0]

# CHECK: vldrw.u32 q0, [r11] @ encoding: [0x9b,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [r11]

# CHECK: vldrw.u32 q3, [r11] @ encoding: [0x9b,0xed,0x00,0x7f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q3, [r11]

# CHECK: vldrw.u32 q0, [r4, #56] @ encoding: [0x94,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [r4, #56]

# CHECK: vldrw.u32 q4, [r4, #56] @ encoding: [0x94,0xed,0x0e,0x9f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q4, [r4, #56]

# CHECK: vldrw.u32 q0, [r8, #56] @ encoding: [0x98,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [r8, #56]

# CHECK: vldrw.u32 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q5, [r4, #56]!

# CHECK: vldrw.u32 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q5, [r4, #56]!

# CHECK: vldrw.u32 q5, [r4], #-28 @ encoding: [0x34,0xec,0x07,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q5, [r4], #-28

# CHECK: vldrw.u32 q5, [r10], #-28 @ encoding: [0x3a,0xec,0x07,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q5, [r10], #-28

# CHECK: vldrw.u32 q5, [sp, #-28] @ encoding: [0x1d,0xed,0x07,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q5, [sp, #-28]

# CHECK: vldrw.u32 q5, [sp, #-64] @ encoding: [0x1d,0xed,0x10,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q5, [sp, #-64]

# CHECK: vldrw.u32 q5, [sp, #-508] @ encoding: [0x1d,0xed,0x7f,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q5, [sp, #-508]

# CHECK: vldrw.u32 q5, [r4, #508]! @ encoding: [0xb4,0xed,0x7f,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q5, [r4, #508]!

# CHECK: vstrw.32 q0, [r0] @ encoding: [0x80,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q0, [r0]

# CHECK: vstrw.32 q1, [r0] @ encoding: [0x80,0xed,0x00,0x3f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q1, [r0]

# CHECK: vstrw.32 q0, [r11] @ encoding: [0x8b,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q0, [r11]

# CHECK: vstrw.32 q3, [r11] @ encoding: [0x8b,0xed,0x00,0x7f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q3, [r11]

# CHECK: vstrw.32 q0, [r4, #56] @ encoding: [0x84,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q0, [r4, #56]

# CHECK: vstrw.32 q4, [r4, #56] @ encoding: [0x84,0xed,0x0e,0x9f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q4, [r4, #56]

# CHECK: vstrw.32 q0, [r8, #56] @ encoding: [0x88,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q0, [r8, #56]

# CHECK: vstrw.32 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [r4, #56]!

# CHECK: vstrw.32 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [r4, #56]!

# CHECK: vstrw.32 q5, [r4], #-28 @ encoding: [0x24,0xec,0x07,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [r4], #-28

# CHECK: vstrw.32 q5, [r10], #-28 @ encoding: [0x2a,0xec,0x07,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [r10], #-28

# CHECK: vstrw.32 q5, [sp, #-28] @ encoding: [0x0d,0xed,0x07,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [sp, #-28]

# CHECK: vstrw.32 q5, [sp, #-64] @ encoding: [0x0d,0xed,0x10,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [sp, #-64]

# CHECK: vstrw.32 q5, [sp, #-508] @ encoding: [0x0d,0xed,0x7f,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [sp, #-508]

# CHECK: vstrw.32 q5, [r4, #508]! @ encoding: [0xa4,0xed,0x7f,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [r4, #508]!

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q0, [r8]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q0, [r8]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [sp, #-64]!

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [sp, #-3]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q5, [sp, #512]

# CHECK: vldrb.u8 q0, [r0, q1] @ encoding: [0x90,0xfc,0x02,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q0, [r0, q1]

# CHECK: vldrb.u8 q3, [r10, q1] @ encoding: [0x9a,0xfc,0x02,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q3, [r10, q1]

# CHECK: vldrb.u16 q0, [r0, q1] @ encoding: [0x90,0xfc,0x82,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q0, [r0, q1]

# CHECK: vldrb.u16 q3, [r9, q1] @ encoding: [0x99,0xfc,0x82,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q3, [r9, q1]

# CHECK: vldrb.s16 q0, [r0, q1] @ encoding: [0x90,0xec,0x82,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q0, [r0, q1]

# CHECK: vldrb.s16 q3, [sp, q1] @ encoding: [0x9d,0xec,0x82,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q3, [sp, q1]

# CHECK: vldrb.u32 q0, [r0, q1] @ encoding: [0x90,0xfc,0x02,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q0, [r0, q1]

# CHECK: vldrb.u32 q3, [r0, q1] @ encoding: [0x90,0xfc,0x02,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q3, [r0, q1]

# CHECK: vldrb.s32 q0, [r0, q1] @ encoding: [0x90,0xec,0x02,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q0, [r0, q1]

# CHECK: vldrb.s32 q3, [r0, q1] @ encoding: [0x90,0xec,0x02,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q3, [r0, q1]

# CHECK: vldrh.u16 q0, [r0, q1] @ encoding: [0x90,0xfc,0x92,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q0, [r0, q1]

# CHECK: vldrh.u16 q3, [r0, q1] @ encoding: [0x90,0xfc,0x92,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q3, [r0, q1]

# CHECK: vldrh.u32 q0, [r0, q1] @ encoding: [0x90,0xfc,0x12,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q0, [r0, q1]

# CHECK: vldrh.u32 q3, [r0, q1] @ encoding: [0x90,0xfc,0x12,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q3, [r0, q1]

# CHECK: vldrh.s32 q0, [r0, q1] @ encoding: [0x90,0xec,0x12,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q0, [r0, q1]

# CHECK: vldrh.s32 q3, [r0, q1] @ encoding: [0x90,0xec,0x12,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q3, [r0, q1]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u8 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u16 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s16 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.u32 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s32 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q0, [r0, q0, uxtw #1]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u32 q0, [r0, q0, uxtw #1]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s32 q0, [r0, q0, uxtw #1]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [r0, q0, uxtw #2]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q0, [r0, q0]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector offset register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q0, [r0, q0, uxtw #3]

# CHECK: vldrh.u16 q0, [r0, q1, uxtw #1] @ encoding: [0x90,0xfc,0x93,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.u16 q0, [r0, q1, uxtw #1]

# CHECK: vldrw.u32 q0, [r0, q1] @ encoding: [0x90,0xfc,0x42,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [r0, q1]

# CHECK: vldrw.u32 q3, [r0, q1] @ encoding: [0x90,0xfc,0x42,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q3, [r0, q1]

# CHECK: vldrw.u32 q0, [r0, q1, uxtw #2] @ encoding: [0x90,0xfc,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [r0, q1, uxtw #2]

# CHECK: vldrw.u32 q0, [sp, q1, uxtw #2] @ encoding: [0x9d,0xfc,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [sp, q1, uxtw #2]

# CHECK: vldrd.u64 q0, [r0, q1] @ encoding: [0x90,0xfc,0xd2,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q0, [r0, q1]

# CHECK: vldrd.u64 q3, [r0, q1] @ encoding: [0x90,0xfc,0xd2,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q3, [r0, q1]

# CHECK: vldrd.u64 q0, [r0, q1, uxtw #3] @ encoding: [0x90,0xfc,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q0, [r0, q1, uxtw #3]

# CHECK: vldrd.u64 q0, [sp, q1, uxtw #3] @ encoding: [0x9d,0xfc,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q0, [sp, q1, uxtw #3]

# CHECK: vstrb.8 q0, [r0, q1] @ encoding: [0x80,0xec,0x02,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q0, [r0, q1]

# CHECK: vstrb.8 q3, [r10, q1] @ encoding: [0x8a,0xec,0x02,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q3, [r10, q1]

# CHECK: vstrb.8 q3, [r0, q3] @ encoding: [0x80,0xec,0x06,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.8 q3, [r0, q3]

# CHECK: vstrb.16 q0, [r0, q1] @ encoding: [0x80,0xec,0x82,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q0, [r0, q1]

# CHECK: vstrb.16 q3, [sp, q1] @ encoding: [0x8d,0xec,0x82,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q3, [sp, q1]

# CHECK: vstrb.16 q3, [r0, q3] @ encoding: [0x80,0xec,0x86,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.16 q3, [r0, q3]

# CHECK: vstrb.32 q0, [r0, q1] @ encoding: [0x80,0xec,0x02,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q0, [r0, q1]

# CHECK: vstrb.32 q3, [r0, q1] @ encoding: [0x80,0xec,0x02,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q3, [r0, q1]

# CHECK: vstrb.32 q3, [r0, q3] @ encoding: [0x80,0xec,0x06,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q3, [r0, q3]

# CHECK: vstrh.16 q0, [r0, q1] @ encoding: [0x80,0xec,0x92,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q0, [r0, q1]

# CHECK: vstrh.16 q3, [r0, q1] @ encoding: [0x80,0xec,0x92,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q3, [r0, q1]

# CHECK: vstrh.16 q3, [r0, q3] @ encoding: [0x80,0xec,0x96,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q3, [r0, q3]

# CHECK: vstrh.32 q0, [r0, q1] @ encoding: [0x80,0xec,0x12,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q0, [r0, q1]

# CHECK: vstrh.32 q3, [r0, q1] @ encoding: [0x80,0xec,0x12,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q3, [r0, q1]

# CHECK: vstrh.32 q3, [r0, q3] @ encoding: [0x80,0xec,0x16,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q3, [r0, q3]

# CHECK: vstrh.16 q0, [r0, q1, uxtw #1] @ encoding: [0x80,0xec,0x93,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q0, [r0, q1, uxtw #1]

# CHECK: vstrh.32 q3, [r8, q3, uxtw #1] @ encoding: [0x88,0xec,0x17,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.32 q3, [r8, q3, uxtw #1]

# CHECK: vstrw.32 q0, [r0, q1] @ encoding: [0x80,0xec,0x42,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q0, [r0, q1]

# CHECK: vstrw.32 q3, [r0, q1] @ encoding: [0x80,0xec,0x42,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q3, [r0, q1]

# CHECK: vstrw.32 q3, [r0, q3] @ encoding: [0x80,0xec,0x46,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q3, [r0, q3]

# CHECK: vstrw.32 q0, [r0, q1, uxtw #2] @ encoding: [0x80,0xec,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q0, [r0, q1, uxtw #2]

# CHECK: vstrw.32 q0, [sp, q1, uxtw #2] @ encoding: [0x8d,0xec,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q0, [sp, q1, uxtw #2]

# CHECK: vstrd.64 q0, [r0, q1] @ encoding: [0x80,0xec,0xd2,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q0, [r0, q1]

# CHECK: vstrd.64 q3, [r0, q1] @ encoding: [0x80,0xec,0xd2,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q3, [r0, q1]

# CHECK: vstrd.64 q3, [r0, q3] @ encoding: [0x80,0xec,0xd6,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q3, [r0, q3]

# CHECK: vstrd.64 q0, [r0, q1, uxtw #3] @ encoding: [0x80,0xec,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q0, [r0, q1, uxtw #3]

# CHECK: vstrd.64 q0, [sp, q1, uxtw #3] @ encoding: [0x8d,0xec,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q0, [sp, q1, uxtw #3]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: operand must be a register in range [q0, q7]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q9, [sp, q1, uxtw #2]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q3, [pc, q1]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q0, [r0, q1, uxtw #1]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q0, [r0, q1, uxtw #1]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q0, [r0, q1, uxtw #2]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q0, [r0, q1, uxtw #2]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [r0, q1, uxtw #1]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.16 q0, [r0, q1, uxtw #2]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.32 q0, [r11, q1, uxtw #1]

# CHECK: vldrw.u32 q0, [q1] @ encoding: [0x92,0xfd,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q0, [q1]

# CHECK: vldrw.u32 q7, [q1] @ encoding: [0x92,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1]

# CHECK: vldrw.u32 q7, [q1]! @ encoding: [0xb2,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1]!

# CHECK: vldrw.u32 q7, [q1, #4] @ encoding: [0x92,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1, #4]

# CHECK: vldrw.u32 q7, [q1, #-4] @ encoding: [0x12,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1, #-4]

# CHECK: vldrw.u32 q7, [q1, #508] @ encoding: [0x92,0xfd,0x7f,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1, #508]

# CHECK: vldrw.u32 q7, [q1, #-508] @ encoding: [0x12,0xfd,0x7f,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1, #-508]

# CHECK: vldrw.u32 q7, [q1, #264] @ encoding: [0x92,0xfd,0x42,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1, #264]

# CHECK: vldrw.u32 q7, [q1, #4]! @ encoding: [0xb2,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1, #4]!

# CHECK: vstrw.32 q0, [q1] @ encoding: [0x82,0xfd,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q0, [q1]

# CHECK: vstrw.32 q1, [q1] @ encoding: [0x82,0xfd,0x00,0x3e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q1, [q1]

# CHECK: vstrw.32 q7, [q1] @ encoding: [0x82,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q7, [q1]

# CHECK: vstrw.32 q7, [q1]! @ encoding: [0xa2,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q7, [q1]!

# CHECK: vstrw.32 q7, [q7] @ encoding: [0x8e,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q7, [q7]

# CHECK: vstrw.32 q7, [q1, #4] @ encoding: [0x82,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q7, [q1, #4]

# CHECK: vstrw.32 q7, [q1, #-4] @ encoding: [0x02,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q7, [q1, #-4]

# CHECK: vstrw.32 q7, [q1, #508] @ encoding: [0x82,0xfd,0x7f,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q7, [q1, #508]

# CHECK: vstrw.32 q7, [q1, #-508] @ encoding: [0x02,0xfd,0x7f,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q7, [q1, #-508]

# CHECK: vstrw.32 q7, [q1, #264]! @ encoding: [0xa2,0xfd,0x42,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q7, [q1, #264]!

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: operand must be a register in range [q0, q7]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q8, [q1]!

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.32 q4, [q1, #3]!

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1, #512]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector pointer register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.32 q1, [q1, #264]

# CHECK: vldrd.u64 q0, [q1] @ encoding: [0x92,0xfd,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q0, [q1]

# CHECK: vldrd.u64 q7, [q1] @ encoding: [0x92,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1]

# CHECK: vldrd.u64 q7, [q1]! @ encoding: [0xb2,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1]!

# CHECK: vldrd.u64 q7, [q1, #8] @ encoding: [0x92,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1, #8]

# CHECK: vldrd.u64 q7, [q1, #-8] @ encoding: [0x12,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1, #-8]

# CHECK: vldrd.u64 q7, [q1, #1016] @ encoding: [0x92,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1, #1016]

# CHECK: vldrd.u64 q7, [q1, #-1016] @ encoding: [0x12,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1, #-1016]

# CHECK: vldrd.u64 q7, [q1, #264] @ encoding: [0x92,0xfd,0x21,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1, #264]

# CHECK: vldrd.u64 q7, [q1, #624] @ encoding: [0x92,0xfd,0x4e,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1, #624]

# CHECK: vldrd.u64 q7, [q1, #264] @ encoding: [0x92,0xfd,0x21,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1, #264]

# CHECK: vldrd.u64 q7, [q1, #-1016]! @ encoding: [0x32,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q7, [q1, #-1016]!

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: destination vector register and vector pointer register can't be identical
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q6, [q6]

# CHECK: vstrd.64 q0, [q1] @ encoding: [0x82,0xfd,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q0, [q1]

# CHECK: vstrd.64 q1, [q1] @ encoding: [0x82,0xfd,0x00,0x3f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q1, [q1]

# CHECK: vstrd.64 q7, [q1] @ encoding: [0x82,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1]

# CHECK: vstrd.64 q7, [q1]! @ encoding: [0xa2,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1]!

# CHECK: vstrd.64 q7, [q7] @ encoding: [0x8e,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q7]

# CHECK: vstrd.64 q7, [q1, #8] @ encoding: [0x82,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1, #8]

# CHECK: vstrd.64 q7, [q1, #-8]! @ encoding: [0x22,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1, #-8]!

# CHECK: vstrd.64 q7, [q1, #1016] @ encoding: [0x82,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1, #1016]

# CHECK: vstrd.64 q7, [q1, #-1016] @ encoding: [0x02,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1, #-1016]

# CHECK: vstrd.64 q7, [q1, #264] @ encoding: [0x82,0xfd,0x21,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1, #264]

# CHECK: vstrd.64 q7, [q1, #624] @ encoding: [0x82,0xfd,0x4e,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1, #624]

# CHECK: vstrd.64 q7, [q1, #264] @ encoding: [0x82,0xfd,0x21,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1, #264]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: operand must be a register in range [q0, q7]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.u64 q8, [q1]!

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q7, [q1, #1024]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q4, [q1, #3]

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.64 q4, [q1, #4]

# CHECK: vldrb.u8 q0, [r0] @ encoding: [0x90,0xed,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s8 q0, [r0]

# CHECK: vldrb.u8 q0, [r0] @ encoding: [0x90,0xed,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.8 q0, [r0]

# CHECK: vldrb.u8 q0, [r8, #56] @ encoding: [0x98,0xed,0x38,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s8 q0, [r8, #56]

# CHECK: vldrb.u8 q0, [r8, #56] @ encoding: [0x98,0xed,0x38,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.8 q0, [r8, #56]

# CHECK: vldrb.u8 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x38,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s8 q5, [r4, #56]!

# CHECK: vldrb.u8 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x38,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.8 q5, [r4, #56]!

# CHECK: vstrb.8 q0, [r0] @ encoding: [0x80,0xed,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.u8 q0, [r0]

# CHECK: vstrb.8 q0, [r0] @ encoding: [0x80,0xed,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.s8 q0, [r0]

# CHECK: vstrb.8 q4, [r4, #56] @ encoding: [0x84,0xed,0x38,0x9e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.u8 q4, [r4, #56]

# CHECK: vstrb.8 q4, [r4, #56] @ encoding: [0x84,0xed,0x38,0x9e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.s8 q4, [r4, #56]

# CHECK: vstrb.8 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x38,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.u8 q5, [r4, #56]!

# CHECK: vstrb.8 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x38,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.s8 q5, [r4, #56]!

# CHECK: vldrh.u16 q0, [r0] @ encoding: [0x90,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s16 q0, [r0]

# CHECK: vldrh.u16 q0, [r0] @ encoding: [0x90,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.f16 q0, [r0]

# CHECK: vldrh.u16 q0, [r0] @ encoding: [0x90,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.16 q0, [r0]

# CHECK: vldrh.u16 q0, [r4, #56] @ encoding: [0x94,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s16 q0, [r4, #56]

# CHECK: vldrh.u16 q0, [r4, #56] @ encoding: [0x94,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.f16 q0, [r4, #56]

# CHECK: vldrh.u16 q0, [r4, #56] @ encoding: [0x94,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.16 q0, [r4, #56]

# CHECK: vldrh.u16 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s16 q5, [r4, #56]!

# CHECK: vldrh.u16 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.f16 q5, [r4, #56]!

# CHECK: vldrh.u16 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.16 q5, [r4, #56]!

# CHECK: vstrh.16 q0, [r0] @ encoding: [0x80,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.u16 q0, [r0]

# CHECK: vstrh.16 q0, [r0] @ encoding: [0x80,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.s16 q0, [r0]

# CHECK: vstrh.16 q0, [r0] @ encoding: [0x80,0xed,0x80,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.f16 q0, [r0]

# CHECK: vstrh.16 q0, [r4, #56] @ encoding: [0x84,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.u16 q0, [r4, #56]

# CHECK: vstrh.16 q0, [r4, #56] @ encoding: [0x84,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.s16 q0, [r4, #56]

# CHECK: vstrh.16 q0, [r4, #56] @ encoding: [0x84,0xed,0x9c,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.f16 q0, [r4, #56]

# CHECK: vstrh.16 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.u16 q5, [r4, #56]!

# CHECK: vstrh.16 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.s16 q5, [r4, #56]!

# CHECK: vstrh.16 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x9c,0xbe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.f16 q5, [r4, #56]!

# CHECK: vldrw.u32 q0, [r0] @ encoding: [0x90,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.s32 q0, [r0]

# CHECK: vldrw.u32 q0, [r0] @ encoding: [0x90,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.f32 q0, [r0]

# CHECK: vldrw.u32 q0, [r0] @ encoding: [0x90,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.32 q0, [r0]

# CHECK: vldrw.u32 q0, [r4, #56] @ encoding: [0x94,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.s32 q0, [r4, #56]

# CHECK: vldrw.u32 q0, [r4, #56] @ encoding: [0x94,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.f32 q0, [r4, #56]

# CHECK: vldrw.u32 q0, [r4, #56] @ encoding: [0x94,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.32 q0, [r4, #56]

# CHECK: vldrw.u32 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.s32 q5, [r4, #56]!

# CHECK: vldrw.u32 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.f32 q5, [r4, #56]!

# CHECK: vldrw.u32 q5, [r4, #56]! @ encoding: [0xb4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.32 q5, [r4, #56]!

# CHECK: vstrw.32 q0, [r0] @ encoding: [0x80,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.u32 q0, [r0]

# CHECK: vstrw.32 q0, [r0] @ encoding: [0x80,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.s32 q0, [r0]

# CHECK: vstrw.32 q0, [r0] @ encoding: [0x80,0xed,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.f32 q0, [r0]

# CHECK: vstrw.32 q0, [r4, #56] @ encoding: [0x84,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.u32 q0, [r4, #56]

# CHECK: vstrw.32 q0, [r4, #56] @ encoding: [0x84,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.s32 q0, [r4, #56]

# CHECK: vstrw.32 q0, [r4, #56] @ encoding: [0x84,0xed,0x0e,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.f32 q0, [r4, #56]

# CHECK: vstrw.32 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.u32 q5, [r4, #56]!

# CHECK: vstrw.32 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.s32 q5, [r4, #56]!

# CHECK: vstrw.32 q5, [r4, #56]! @ encoding: [0xa4,0xed,0x0e,0xbf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.f32 q5, [r4, #56]!

# CHECK: vldrb.u8 q0, [r0, q1] @ encoding: [0x90,0xfc,0x02,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.s8 q0, [r0, q1]

# CHECK: vldrb.u8 q0, [r0, q1] @ encoding: [0x90,0xfc,0x02,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrb.8 q0, [r0, q1]

# CHECK: vldrh.u16 q3, [r0, q1] @ encoding: [0x90,0xfc,0x92,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s16 q3, [r0, q1]

# CHECK: vldrh.u16 q3, [r0, q1] @ encoding: [0x90,0xfc,0x92,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.f16 q3, [r0, q1]

# CHECK: vldrh.u16 q3, [r0, q1] @ encoding: [0x90,0xfc,0x92,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.16 q3, [r0, q1]

# CHECK: vldrh.u16 q0, [r0, q1, uxtw #1] @ encoding: [0x90,0xfc,0x93,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.s16 q0, [r0, q1, uxtw #1]

# CHECK: vldrh.u16 q0, [r0, q1, uxtw #1] @ encoding: [0x90,0xfc,0x93,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.f16 q0, [r0, q1, uxtw #1]

# CHECK: vldrh.u16 q0, [r0, q1, uxtw #1] @ encoding: [0x90,0xfc,0x93,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrh.16 q0, [r0, q1, uxtw #1]

# CHECK: vldrw.u32 q0, [r0, q1] @ encoding: [0x90,0xfc,0x42,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.s32 q0, [r0, q1]

# CHECK: vldrw.u32 q0, [r0, q1] @ encoding: [0x90,0xfc,0x42,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.f32 q0, [r0, q1]

# CHECK: vldrw.u32 q0, [r0, q1] @ encoding: [0x90,0xfc,0x42,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.32 q0, [r0, q1]

# CHECK: vldrw.u32 q0, [r0, q1, uxtw #2] @ encoding: [0x90,0xfc,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.s32 q0, [r0, q1, uxtw #2]

# CHECK: vldrw.u32 q0, [r0, q1, uxtw #2] @ encoding: [0x90,0xfc,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.f32 q0, [r0, q1, uxtw #2]

# CHECK: vldrw.u32 q0, [r0, q1, uxtw #2] @ encoding: [0x90,0xfc,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.32 q0, [r0, q1, uxtw #2]

# CHECK: vldrd.u64 q0, [r0, q1] @ encoding: [0x90,0xfc,0xd2,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.s64 q0, [r0, q1]

# CHECK: vldrd.u64 q0, [r0, q1] @ encoding: [0x90,0xfc,0xd2,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.f64 q0, [r0, q1]

# CHECK: vldrd.u64 q0, [r0, q1] @ encoding: [0x90,0xfc,0xd2,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.64 q0, [r0, q1]

# CHECK: vldrd.u64 q0, [r0, q1, uxtw #3] @ encoding: [0x90,0xfc,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.s64 q0, [r0, q1, uxtw #3]

# CHECK: vldrd.u64 q0, [r0, q1, uxtw #3] @ encoding: [0x90,0xfc,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.f64 q0, [r0, q1, uxtw #3]

# CHECK: vldrd.u64 q0, [r0, q1, uxtw #3] @ encoding: [0x90,0xfc,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.64 q0, [r0, q1, uxtw #3]

# CHECK: vstrb.8 q0, [r0, q1] @ encoding: [0x80,0xec,0x02,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.u8 q0, [r0, q1]

# CHECK: vstrb.8 q0, [r0, q1] @ encoding: [0x80,0xec,0x02,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrb.s8 q0, [r0, q1]

# CHECK: vstrh.16 q3, [r0, q1] @ encoding: [0x80,0xec,0x92,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.u16 q3, [r0, q1]

# CHECK: vstrh.16 q3, [r0, q1] @ encoding: [0x80,0xec,0x92,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.s16 q3, [r0, q1]

# CHECK: vstrh.16 q3, [r0, q1] @ encoding: [0x80,0xec,0x92,0x6e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.f16 q3, [r0, q1]

# CHECK: vstrh.16 q0, [r0, q1, uxtw #1] @ encoding: [0x80,0xec,0x93,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.u16 q0, [r0, q1, uxtw #1]

# CHECK: vstrh.16 q0, [r0, q1, uxtw #1] @ encoding: [0x80,0xec,0x93,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.s16 q0, [r0, q1, uxtw #1]

# CHECK: vstrh.16 q0, [r0, q1, uxtw #1] @ encoding: [0x80,0xec,0x93,0x0e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrh.f16 q0, [r0, q1, uxtw #1]

# CHECK: vstrw.32 q0, [r0, q1] @ encoding: [0x80,0xec,0x42,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.u32 q0, [r0, q1]

# CHECK: vstrw.32 q0, [r0, q1] @ encoding: [0x80,0xec,0x42,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.s32 q0, [r0, q1]

# CHECK: vstrw.32 q0, [r0, q1] @ encoding: [0x80,0xec,0x42,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.f32 q0, [r0, q1]

# CHECK: vstrw.32 q0, [r0, q1, uxtw #2] @ encoding: [0x80,0xec,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.u32 q0, [r0, q1, uxtw #2]

# CHECK: vstrw.32 q0, [r0, q1, uxtw #2] @ encoding: [0x80,0xec,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.s32 q0, [r0, q1, uxtw #2]

# CHECK: vstrw.32 q0, [r0, q1, uxtw #2] @ encoding: [0x80,0xec,0x43,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.f32 q0, [r0, q1, uxtw #2]

# CHECK: vstrd.64 q3, [r0, q1] @ encoding: [0x80,0xec,0xd2,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.u64 q3, [r0, q1]

# CHECK: vstrd.64 q3, [r0, q1] @ encoding: [0x80,0xec,0xd2,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.s64 q3, [r0, q1]

# CHECK: vstrd.64 q3, [r0, q1] @ encoding: [0x80,0xec,0xd2,0x6f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.f64 q3, [r0, q1]

# CHECK: vstrd.64 q0, [r0, q1, uxtw #3] @ encoding: [0x80,0xec,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.u64 q0, [r0, q1, uxtw #3]

# CHECK: vstrd.64 q0, [r0, q1, uxtw #3] @ encoding: [0x80,0xec,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.s64 q0, [r0, q1, uxtw #3]

# CHECK: vstrd.64 q0, [r0, q1, uxtw #3] @ encoding: [0x80,0xec,0xd3,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.f64 q0, [r0, q1, uxtw #3]

# CHECK: vldrw.u32 q0, [q1] @ encoding: [0x92,0xfd,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.s32 q0, [q1]

# CHECK: vldrw.u32 q0, [q1] @ encoding: [0x92,0xfd,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.f32 q0, [q1]

# CHECK: vldrw.u32 q0, [q1] @ encoding: [0x92,0xfd,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.32 q0, [q1]

# CHECK: vldrw.u32 q7, [q1]! @ encoding: [0xb2,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.s32 q7, [q1]!

# CHECK: vldrw.u32 q7, [q1]! @ encoding: [0xb2,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.f32 q7, [q1]!

# CHECK: vldrw.u32 q7, [q1]! @ encoding: [0xb2,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.32 q7, [q1]!

# CHECK: vldrw.u32 q7, [q1, #4] @ encoding: [0x92,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.s32 q7, [q1, #4]

# CHECK: vldrw.u32 q7, [q1, #4] @ encoding: [0x92,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.f32 q7, [q1, #4]

# CHECK: vldrw.u32 q7, [q1, #4] @ encoding: [0x92,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.32 q7, [q1, #4]

# CHECK: vldrw.u32 q7, [q1, #4]! @ encoding: [0xb2,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.s32 q7, [q1, #4]!

# CHECK: vldrw.u32 q7, [q1, #4]! @ encoding: [0xb2,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.f32 q7, [q1, #4]!

# CHECK: vldrw.u32 q7, [q1, #4]! @ encoding: [0xb2,0xfd,0x01,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrw.u32 q7, [q1, #4]!

# CHECK: vstrw.32 q0, [q1] @ encoding: [0x82,0xfd,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.u32 q0, [q1]

# CHECK: vstrw.32 q0, [q1] @ encoding: [0x82,0xfd,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.s32 q0, [q1]

# CHECK: vstrw.32 q0, [q1] @ encoding: [0x82,0xfd,0x00,0x1e]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.f32 q0, [q1]

# CHECK: vstrw.32 q7, [q1]! @ encoding: [0xa2,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.u32 q7, [q1]!

# CHECK: vstrw.32 q7, [q1]! @ encoding: [0xa2,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.s32 q7, [q1]!

# CHECK: vstrw.32 q7, [q1]! @ encoding: [0xa2,0xfd,0x00,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.f32 q7, [q1]!

# CHECK: vstrw.32 q7, [q1, #508] @ encoding: [0x82,0xfd,0x7f,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.u32 q7, [q1, #508]

# CHECK: vstrw.32 q7, [q1, #508] @ encoding: [0x82,0xfd,0x7f,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.s32 q7, [q1, #508]

# CHECK: vstrw.32 q7, [q1, #508] @ encoding: [0x82,0xfd,0x7f,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.f32 q7, [q1, #508]

# CHECK: vstrw.32 q7, [q1, #264]! @ encoding: [0xa2,0xfd,0x42,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.u32 q7, [q1, #264]!

# CHECK: vstrw.32 q7, [q1, #264]! @ encoding: [0xa2,0xfd,0x42,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.s32 q7, [q1, #264]!

# CHECK: vstrw.32 q7, [q1, #264]! @ encoding: [0xa2,0xfd,0x42,0xfe]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrw.f32 q7, [q1, #264]!

# CHECK: vldrd.u64 q0, [q1] @ encoding: [0x92,0xfd,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.s64 q0, [q1]

# CHECK: vldrd.u64 q0, [q1] @ encoding: [0x92,0xfd,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.f64 q0, [q1]

# CHECK: vldrd.u64 q0, [q1] @ encoding: [0x92,0xfd,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.64 q0, [q1]

# CHECK: vldrd.u64 q7, [q1]! @ encoding: [0xb2,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.s64 q7, [q1]!

# CHECK: vldrd.u64 q7, [q1]! @ encoding: [0xb2,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.f64 q7, [q1]!

# CHECK: vldrd.u64 q7, [q1]! @ encoding: [0xb2,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.64 q7, [q1]!

# CHECK: vldrd.u64 q7, [q1, #8] @ encoding: [0x92,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.s64 q7, [q1, #8]

# CHECK: vldrd.u64 q7, [q1, #8] @ encoding: [0x92,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.f64 q7, [q1, #8]

# CHECK: vldrd.u64 q7, [q1, #8] @ encoding: [0x92,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.64 q7, [q1, #8]

# CHECK: vldrd.u64 q7, [q1, #-1016]! @ encoding: [0x32,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.s64 q7, [q1, #-1016]!

# CHECK: vldrd.u64 q7, [q1, #-1016]! @ encoding: [0x32,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.f64 q7, [q1, #-1016]!

# CHECK: vldrd.u64 q7, [q1, #-1016]! @ encoding: [0x32,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vldrd.64 q7, [q1, #-1016]!

# CHECK: vstrd.64 q0, [q1] @ encoding: [0x82,0xfd,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.u64 q0, [q1]

# CHECK: vstrd.64 q0, [q1] @ encoding: [0x82,0xfd,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.s64 q0, [q1]

# CHECK: vstrd.64 q0, [q1] @ encoding: [0x82,0xfd,0x00,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.f64 q0, [q1]

# CHECK: vstrd.64 q7, [q1]! @ encoding: [0xa2,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.u64 q7, [q1]!

# CHECK: vstrd.64 q7, [q1]! @ encoding: [0xa2,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.s64 q7, [q1]!

# CHECK: vstrd.64 q7, [q1]! @ encoding: [0xa2,0xfd,0x00,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.f64 q7, [q1]!

# CHECK: vstrd.64 q7, [q1, #1016] @ encoding: [0x82,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.u64 q7, [q1, #1016]

# CHECK: vstrd.64 q7, [q1, #1016] @ encoding: [0x82,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.s64 q7, [q1, #1016]

# CHECK: vstrd.64 q7, [q1, #1016] @ encoding: [0x82,0xfd,0x7f,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.f64 q7, [q1, #1016]

# CHECK: vstrd.64 q7, [q1, #-8]! @ encoding: [0x22,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.u64 q7, [q1, #-8]!

# CHECK: vstrd.64 q7, [q1, #-8]! @ encoding: [0x22,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.s64 q7, [q1, #-8]!

# CHECK: vstrd.64 q7, [q1, #-8]! @ encoding: [0x22,0xfd,0x01,0xff]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vstrd.f64 q7, [q1, #-8]!

vpste
vstrwt.f32 q7, [q1, #264]!
vldrde.64 q7, [q1, #8]
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vstrwt.32 q7, [q1, #264]! @ encoding: [0xa2,0xfd,0x42,0xfe]
# CHECK: vldrde.u64 q7, [q1, #8] @ encoding: [0x92,0xfd,0x01,0xff]
