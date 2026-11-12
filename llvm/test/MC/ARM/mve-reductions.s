# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s

# CHECK: vabav.s8  r0, q1, q3 @ encoding: [0x82,0xee,0x07,0x0f]
vabav.s8 r0, q1, q3

# CHECK: vabav.s16  r0, q1, q3 @ encoding: [0x92,0xee,0x07,0x0f]
vabav.s16 r0, q1, q3

# CHECK: vabav.s32  r0, q1, q3 @ encoding: [0xa2,0xee,0x07,0x0f]
vabav.s32 r0, q1, q3

# CHECK: vabav.u8  r0, q1, q3 @ encoding: [0x82,0xfe,0x07,0x0f]
vabav.u8 r0, q1, q3

# CHECK: vabav.u16  r0, q1, q3 @ encoding: [0x92,0xfe,0x07,0x0f]
vabav.u16 r0, q1, q3

# CHECK: vabav.u32  r0, q1, q3 @ encoding: [0xa2,0xfe,0x07,0x0f]
vabav.u32 r0, q1, q3

# CHECK: vaddv.s16 lr, q0  @ encoding: [0xf5,0xee,0x00,0xef]
vaddv.s16 lr, q0

# ERROR: [[@LINE+1]]:11: {{error|note}}: operand must be an even-numbered register
vaddv.s16 r1, q0

# CHECK: vpte.i8 eq, q0, q0
# CHECK: vaddvt.s16 r0, q6  @ encoding: [0xf5,0xee,0x0c,0x0f]
# CHECK: vaddve.s16 r0, q6  @ encoding: [0xf5,0xee,0x0c,0x0f]
vpte.i8 eq, q0, q0
vaddvt.s16 r0, q6
vaddve.s16 r0, q6

# CHECK: vaddva.s16 lr, q0  @ encoding: [0xf5,0xee,0x20,0xef]
vaddva.s16 lr, q0

# CHECK: vpte.i8 eq, q0, q0
# CHECK: vaddvat.s16 lr, q0  @ encoding: [0xf5,0xee,0x20,0xef]
# CHECK: vaddvae.s16 lr, q0  @ encoding: [0xf5,0xee,0x20,0xef]
vpte.i8 eq, q0, q0
vaddvat.s16 lr, q0
vaddvae.s16 lr, q0

# CHECK: vaddlv.s32 r0, r9, q2  @ encoding: [0xc9,0xee,0x04,0x0f]
vaddlv.s32 r0, r9, q2

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an odd-numbered register in range [r1,r11]
vaddlv.s32 r0, r2, q2

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an even-numbered register
vaddlv.s32 r1, r3, q2

# CHECK: vaddlv.u32 r0, r1, q1  @ encoding: [0x89,0xfe,0x02,0x0f]
vaddlv.u32 r0, r1, q1

# CHECK: vminv.s8 lr, q0  @ encoding: [0xe2,0xee,0x80,0xef]
vminv.s8 lr, q0

# CHECK: vminv.s16 lr, q0  @ encoding: [0xe6,0xee,0x80,0xef]
vminv.s16 lr, q0

# CHECK: vminv.s32 lr, q2  @ encoding: [0xea,0xee,0x84,0xef]
vminv.s32 lr, q2

# CHECK: vminv.u8 r0, q0  @ encoding: [0xe2,0xfe,0x80,0x0f]
vminv.u8 r0, q0

# CHECK: vminv.u32 r10, q3  @ encoding: [0xea,0xfe,0x86,0xaf]
vminv.u32 r10, q3

# CHECK: vminav.s16 r0, q0  @ encoding: [0xe4,0xee,0x80,0x0f]
vminav.s16 r0, q0

# CHECK: vminav.s8 r0, q1  @ encoding: [0xe0,0xee,0x82,0x0f]
vminav.s8 r0, q1

# CHECK: vminav.s32 lr, q1  @ encoding: [0xe8,0xee,0x82,0xef]
vminav.s32 lr, q1

# CHECK: vmaxv.s8 lr, q4  @ encoding: [0xe2,0xee,0x08,0xef]
vmaxv.s8 lr, q4

# CHECK: vmaxv.s16 lr, q0  @ encoding: [0xe6,0xee,0x00,0xef]
vmaxv.s16 lr, q0

# CHECK: vmaxv.s32 r1, q1  @ encoding: [0xea,0xee,0x02,0x1f]
vmaxv.s32 r1, q1

# CHECK: vmaxv.u8 r0, q4  @ encoding: [0xe2,0xfe,0x08,0x0f]
vmaxv.u8 r0, q4

# CHECK: vmaxv.u16 r0, q1  @ encoding: [0xe6,0xfe,0x02,0x0f]
vmaxv.u16 r0, q1

# CHECK: vmaxv.u32 r1, q0  @ encoding: [0xea,0xfe,0x00,0x1f]
vmaxv.u32 r1, q0

# CHECK: vmaxav.s8 lr, q6  @ encoding: [0xe0,0xee,0x0c,0xef]
vmaxav.s8 lr, q6

# CHECK: vmaxav.s16 r0, q6  @ encoding: [0xe4,0xee,0x0c,0x0f]
vmaxav.s16 r0, q6

# CHECK: vmaxav.s32 r10, q7  @ encoding: [0xe8,0xee,0x0e,0xaf]
vmaxav.s32 r10, q7

# CHECK: vmlav.s16 lr, q0, q7  @ encoding: [0xf0,0xee,0x0e,0xee]
vmladav.s16 lr, q0, q7

# CHECK: vmlav.s32 lr, q0, q4  @ encoding: [0xf1,0xee,0x08,0xee]
vmladav.s32 lr, q0, q4

# CHECK: vmlav.u16 lr, q0, q7  @ encoding: [0xf0,0xfe,0x0e,0xee]
vmladav.u16 lr, q0, q7

# CHECK: vmlav.u32 lr, q0, q0  @ encoding: [0xf1,0xfe,0x00,0xee]
vmladav.u32 lr, q0, q0

# CHECK: vmlava.s16 lr, q0, q4  @ encoding: [0xf0,0xee,0x28,0xee]
vmladava.s16 lr, q0, q4

# CHECK: vmladavx.s16 r0, q0, q7  @ encoding: [0xf0,0xee,0x0e,0x1e]
vmladavx.s16 r0, q0, q7

# CHECK: vmladavax.s16 lr, q0, q7  @ encoding: [0xf0,0xee,0x2e,0xfe]
vmladavax.s16 lr, q0, q7

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmladavax.u16 r0, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmladavx.u16 r0, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmladavax.u32 r0, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmladavx.u32 r0, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmladavax.u8 r0, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmladavx.u8 r0, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmlaldavax.u16 r2, r3, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmlaldavx.u16 r2, r3, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmlaldavax.u32 r2, r3, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmlaldavx.u32 r2, r3, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vrmlaldavhax.u32 r2, r3, q4, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vrmlaldavhx.u32 r2, r3, q4, q5

# CHECK: vmlav.s8 lr, q3, q0  @ encoding: [0xf6,0xee,0x00,0xef]
vmladav.s8 lr, q3, q0

# CHECK: vmlav.u8 lr, q1, q7  @ encoding: [0xf2,0xfe,0x0e,0xef]
vmladav.u8 lr, q1, q7

# CHECK: vrmlalvh.s32 lr, r1, q6, q2  @ encoding: [0x8c,0xee,0x04,0xef]
vrmlaldavh.s32 lr, r1, q6, q2

# CHECK: vrmlalvh.u32 lr, r1, q5, q2  @ encoding: [0x8a,0xfe,0x04,0xef]
vrmlaldavh.u32 lr, r1, q5, q2

# CHECK: vrmlalvh.u32 lr, r1, q5, q2  @ encoding: [0x8a,0xfe,0x04,0xef]
vrmlaldavh.u32 lr, r1, q5, q2

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an even-numbered register
vrmlaldavh.u32 r1, r3, q5, q2

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an odd-numbered register in range [r1,r11]
vrmlaldavh.u32 r2, r4, q5, q2

# CHECK: vrmlaldavhax.s32 lr, r1, q3, q0  @ encoding: [0x86,0xee,0x20,0xff]
vrmlaldavhax.s32 lr, r1, q3, q0

# CHECK: vrmlsldavh.s32 lr, r11, q6, q5  @ encoding: [0xdc,0xfe,0x0b,0xee]
vrmlsldavh.s32 lr, r11, q6, q5

# CHECK: vmlsdav.s16 lr, q0, q3  @ encoding: [0xf0,0xee,0x07,0xee]
vmlsdav.s16 lr, q0, q3

# CHECK: vrmlalvh.s32 lr, r1, q6, q2  @ encoding: [0x8c,0xee,0x04,0xef]
vrmlalvh.s32 lr, r1, q6, q2

# CHECK: vrmlalvh.u32 lr, r1, q5, q2  @ encoding: [0x8a,0xfe,0x04,0xef]
vrmlalvh.u32 lr, r1, q5, q2

# CHECK: vrmlalvha.s32 lr, r1, q3, q6  @ encoding: [0x86,0xee,0x2c,0xef]
vrmlalvha.s32 lr, r1, q3, q6

# CHECK: vrmlalvha.u32 lr, r1, q7, q1  @ encoding: [0x8e,0xfe,0x22,0xef]
vrmlalvha.u32 lr, r1, q7, q1

# CHECK: vmlsdav.s16 lr, q0, q3  @ encoding: [0xf0,0xee,0x07,0xee]
vmlsdav.s16 lr, q0, q3

# CHECK: vmlsdav.s32 lr, q2, q6  @ encoding: [0xf5,0xee,0x0d,0xee]
vmlsdav.s32 lr, q2, q6

# CHECK: vpte.i8 eq, q0, q0
# CHECK: vmlsdavaxt.s16 lr, q1, q4  @ encoding: [0xf2,0xee,0x29,0xfe]
# CHECK: vmlsdavaxe.s16 lr, q1, q4  @ encoding: [0xf2,0xee,0x29,0xfe]
vpte.i8 eq, q0, q0
vmlsdavaxt.s16 lr, q1, q4
vmlsdavaxe.s16 lr, q1, q4

# CHECK: vmlav.s16 lr, q0, q7  @ encoding: [0xf0,0xee,0x0e,0xee]
vmlav.s16 lr, q0, q7

# CHECK: vmlalv.s16 lr, r1, q4, q1  @ encoding: [0x88,0xee,0x02,0xee]
vmlaldav.s16 lr, r1, q4, q1

# CHECK: vmlalv.s32 lr, r11, q4, q1  @ encoding: [0xd9,0xee,0x02,0xee]
vmlaldav.s32 lr, r11, q4, q1

# CHECK: vmlalv.s32 r0, r1, q7, q6  @ encoding: [0x8f,0xee,0x0c,0x0e]
vmlalv.s32 r0, r1, q7, q6

# CHECK: vmlalv.u16 lr, r11, q5, q4  @ encoding: [0xda,0xfe,0x08,0xee]
vmlalv.u16 lr, r11, q5, q4
