// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck %s --check-prefix=VI
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOVI


.globl global
.globl gds

// Parse a global expression
s_mov_b32 s0, global
// VI: s_mov_b32 s0, global ; encoding: [0xff,0x00,0x80,0xbe,A,A,A,A]
// VI-NEXT: ;   fixup A - offset: 4, value: global, kind: FK_PCRel_4

// Use a token with the same name as a global
ds_gws_init v2 gds
// VI: ds_gws_init v2 gds ; encoding: [0x00,0x00,0x33,0xd9,0x00,0x02,0x00,0x00]

// Use a global with the same name as a token
s_mov_b32 s0, gds
// VI: s_mov_b32 s0, gds ; encoding: [0xff,0x00,0x80,0xbe,A,A,A,A]
// VI-NEXT: ;   fixup A - offset: 4, value: gds, kind: FK_PCRel_4

// Use a binary expression
s_mov_b32 s0, gds+4
// VI: s_mov_b32 s0, gds+4 ; encoding: [0xff,0x00,0x80,0xbe,A,A,A,A]
// VI-NEXT: ;   fixup A - offset: 4, value: gds+4, kind: FK_PCRel_4

// Consecutive instructions with no blank line in between to make sure we
// don't call Lex() too many times.
s_add_u32 s0, s0, global+4
s_addc_u32 s1, s1, 0
// VI: s_add_u32 s0, s0, global+4
// VI: s_addc_u32 s1, s1, 0

// Use a computed expression that results in an inline immediate.
.set foo, 4
s_mov_b32 s0, foo+2
// VI: s_mov_b32 s0, 6 ; encoding: [0x86,0x00,0x80,0xbe]

// Use a computed expression that results in a non-inline immediate.
.set foo, 512
s_mov_b32 s0, foo+2
// VI: s_mov_b32 s0, 514 ; encoding: [0xff,0x00,0x80,0xbe,0x02,0x02,0x00,0x00]

v_mul_f32 v0, foo+2, v2
// VI: v_mul_f32_e32 v0, 514, v2 ; encoding: [0xff,0x04,0x00,0x0a,0x02,0x02,0x00,0x00]

BB1:
v_nop_e64
BB2:
s_sub_u32 vcc_lo, vcc_lo, (BB2+4)-BB1
// VI: s_sub_u32 vcc_lo, vcc_lo, (BB2+4)-BB1 ; encoding: [0x6a,0xff,0xea,0x80,A,A,A,A]
// VI-NEXT: ;   fixup A - offset: 4, value: (BB2+4)-BB1, kind: FK_Data_4

t=1
s_sub_u32 s0, s0, -t
// VI: s_sub_u32 s0, s0, -1            ; encoding: [0x00,0xc1,0x80,0x80]

t=-1
s_sub_u32 s0, s0, -t
// VI: s_sub_u32 s0, s0, 1             ; encoding: [0x00,0x81,0x80,0x80]

s_sub_u32 s0, s0, -2+1
// VI: s_sub_u32 s0, s0, -1            ; encoding: [0x00,0xc1,0x80,0x80]

t=1
s_sub_u32 s0, s0, -2+t
// VI: s_sub_u32 s0, s0, -1            ; encoding: [0x00,0xc1,0x80,0x80]

s_sub_u32 s0, s0, -1.0 + 10000000000
// NOVI: error: invalid operand for instruction

t=10000000000
s_sub_u32 s0, s0, 1.0 + t
// NOVI: error: invalid operand for instruction
