// RUN: llvm-mc -triple thumbv8.1m.main-arm-none-eabi -mattr=+pacbti %s -show-encoding -o - | FileCheck %s
// RUN: not llvm-mc -triple thumbv8.1m.main-arm-none-eabi -mattr=-pacbti %s -show-encoding -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NOPACBTI

// CHECK: autg	r0, r1, r2              @ encoding: [0x51,0xfb,0x02,0x0f]
// CHECK-NOPACBTI: error: instruction requires: pacbti
autg r0, r1, r2
// CHECK: autg	r12, lr, sp             @ encoding: [0x5e,0xfb,0x0d,0xcf]
// CHECK-NOPACBTI: error: instruction requires: pacbti
autg ip, lr, sp
// CHECK: aut	r12, lr, sp               @ encoding: [0xaf,0xf3,0x2d,0x80]
aut ip, lr, sp
// CHECK: aut	r12,lr,sp               @ encoding: [0xaf,0xf3,0x2d,0x80]
hint.w #45
// CHECK: bxaut	r0, r1, r2              @ encoding: [0x51,0xfb,0x12,0x0f]
// CHECK-NOPACBTI: error: instruction requires: pacbti
bxaut r0, r1, r2

// CHECK: bti	                          @ encoding: [0xaf,0xf3,0x0f,0x80]
bti
// CHECK: bti	                          @ encoding: [0xaf,0xf3,0x0f,0x80]
hint.w #15

// CHECK: pacg r0, r1, r2               @ encoding: [0x61,0xfb,0x02,0xf0]
// CHECK-NOPACBTI: error: instruction requires: pacbti
pacg r0, r1, r2
// CHECK: pacg r12, lr, sp              @ encoding: [0x6e,0xfb,0x0d,0xfc]
// CHECK-NOPACBTI: error: instruction requires: pacbti
pacg ip, lr, sp
// CHECK: pac r12, lr, sp               @ encoding: [0xaf,0xf3,0x1d,0x80]
pac ip, lr, sp
// CHECK: pac r12,lr,sp               @ encoding: [0xaf,0xf3,0x1d,0x80]
hint.w #29
// CHECK: pacbti r12, lr, sp            @ encoding: [0xaf,0xf3,0x0d,0x80]
pacbti ip, lr, sp
// CHECK: pacbti r12,lr,sp            @ encoding: [0xaf,0xf3,0x0d,0x80]
hint.w #13

// CHECK: msr	pac_key_p_0, r0           @ encoding: [0x80,0xf3,0x20,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_p_0, r0
// CHECK: msr	pac_key_p_1, r0           @ encoding: [0x80,0xf3,0x21,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_p_1, r0
// CHECK: msr	pac_key_p_2, r0           @ encoding: [0x80,0xf3,0x22,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_p_2, r0
// CHECK: msr	pac_key_p_3, r0           @ encoding: [0x80,0xf3,0x23,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_p_3, r0
// CHECK: msr	pac_key_u_0, r0           @ encoding: [0x80,0xf3,0x24,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_u_0, r0
// CHECK: msr	pac_key_u_1, r0           @ encoding: [0x80,0xf3,0x25,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_u_1, r0
// CHECK: msr	pac_key_u_2, r0           @ encoding: [0x80,0xf3,0x26,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_u_2, r0
// CHECK: msr	pac_key_u_3, r0           @ encoding: [0x80,0xf3,0x27,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_u_3, r0
// CHECK: msr	pac_key_p_0_ns, r0        @ encoding: [0x80,0xf3,0xa0,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_p_0_ns, r0
// CHECK: msr	pac_key_p_1_ns, r0        @ encoding: [0x80,0xf3,0xa1,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_p_1_ns, r0
// CHECK: msr	pac_key_p_2_ns, r0        @ encoding: [0x80,0xf3,0xa2,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_p_2_ns, r0
// CHECK: msr	pac_key_p_3_ns, r0        @ encoding: [0x80,0xf3,0xa3,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_p_3_ns, r0
// CHECK: msr	pac_key_u_0_ns, r0        @ encoding: [0x80,0xf3,0xa4,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_u_0_ns, r0
// CHECK: msr	pac_key_u_1_ns, r0        @ encoding: [0x80,0xf3,0xa5,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_u_1_ns, r0
// CHECK: msr	pac_key_u_2_ns, r0        @ encoding: [0x80,0xf3,0xa6,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_u_2_ns, r0
// CHECK: msr	pac_key_u_3_ns, r0        @ encoding: [0x80,0xf3,0xa7,0x88]
// CHECK-NOPACBTI: error: invalid operand for instruction
msr pac_key_u_3_ns, r0

// CHECK: mrs	r0, pac_key_p_0           @ encoding: [0xef,0xf3,0x20,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_p_0
// CHECK: mrs	r0, pac_key_p_1           @ encoding: [0xef,0xf3,0x21,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_p_1
// CHECK: mrs	r0, pac_key_p_2           @ encoding: [0xef,0xf3,0x22,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_p_2
// CHECK: mrs	r0, pac_key_p_3           @ encoding: [0xef,0xf3,0x23,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_p_3
// CHECK: mrs	r0, pac_key_u_0           @ encoding: [0xef,0xf3,0x24,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_u_0
// CHECK: mrs	r0, pac_key_u_1           @ encoding: [0xef,0xf3,0x25,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_u_1
// CHECK: mrs	r0, pac_key_u_2           @ encoding: [0xef,0xf3,0x26,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_u_2
// CHECK: mrs	r0, pac_key_u_3           @ encoding: [0xef,0xf3,0x27,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_u_3
// CHECK: mrs	r0, pac_key_p_0_ns        @ encoding: [0xef,0xf3,0xa0,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_p_0_ns
// CHECK: mrs	r0, pac_key_p_1_ns        @ encoding: [0xef,0xf3,0xa1,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_p_1_ns
// CHECK: mrs	r0, pac_key_p_2_ns        @ encoding: [0xef,0xf3,0xa2,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_p_2_ns
// CHECK: mrs	r0, pac_key_p_3_ns        @ encoding: [0xef,0xf3,0xa3,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_p_3_ns
// CHECK: mrs	r0, pac_key_u_0_ns        @ encoding: [0xef,0xf3,0xa4,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_u_0_ns
// CHECK: mrs	r0, pac_key_u_1_ns        @ encoding: [0xef,0xf3,0xa5,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_u_1_ns
// CHECK: mrs	r0, pac_key_u_2_ns        @ encoding: [0xef,0xf3,0xa6,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_u_2_ns
// CHECK: mrs	r0, pac_key_u_3_ns        @ encoding: [0xef,0xf3,0xa7,0x80]
// CHECK-NOPACBTI: error: invalid operand for instruction
mrs r0, pac_key_u_3_ns
