// RUN: not llvm-mc -triple=amdgcn -mcpu=fiji -show-encoding %s | FileCheck %s --check-prefix=VI
// RUN: not llvm-mc -triple=amdgcn -mcpu=fiji %s 2>&1 | FileCheck %s --check-prefix=NOVI --implicit-check-not=error:

//===----------------------------------------------------------------------===//
// Floating-point expressions are not supported
//===----------------------------------------------------------------------===//

s_sub_u32 s0, s0, -1.0 + 10000000000
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

t=10000000000
s_sub_u32 s0, s0, 1.0 + t
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f32 v1, 1.0 + 1.0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_ceil_f32 v1, -1.0 + 1.0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

//===----------------------------------------------------------------------===//
// Constant expressions may be used with SP3 'abs' modifiers |...|
// These expressions must be primary expressions to avoid incorrect
// interpretation of closing "|".
//===----------------------------------------------------------------------===//

i1=1
fm1=0xBF800000 // -1.0f
hm1=0xBC00     // -1.0h

v_ceil_f32 v1, |i1|
// VI: v_ceil_f32_e32 v1, 1            ; encoding: [0x81,0x3a,0x02,0x7e]

v_ceil_f32 v1, |(i1+1)|
// VI: v_ceil_f32_e32 v1, 2            ; encoding: [0x82,0x3a,0x02,0x7e]

v_ceil_f32 v1, |-(i1+1)|
// VI: v_ceil_f32_e32 v1, 0x7ffffffe   ; encoding: [0xff,0x3a,0x02,0x7e,0xfe,0xff,0xff,0x7f]

v_ceil_f32 v1, |fm1|
// VI: v_ceil_f32_e32 v1, 1.0          ; encoding: [0xf2,0x3a,0x02,0x7e]

v_mad_f16 v5, v1, v2, |i1|
// VI: v_mad_f16 v5, v1, v2, |1|       ; encoding: [0x05,0x04,0xea,0xd1,0x01,0x05,0x06,0x02]

v_mad_f16 v5, v1, v2, |(i1+1)|
// VI: v_mad_f16 v5, v1, v2, |2|       ; encoding: [0x05,0x04,0xea,0xd1,0x01,0x05,0x0a,0x02]

v_mad_f16 v5, v1, v2, |hm1|
// VI: v_mad_f16 v5, v1, v2, |-1.0|    ; encoding: [0x05,0x04,0xea,0xd1,0x01,0x05,0xce,0x03]

// Only primary expressions are allowed

v_ceil_f32 v1, |1+i1|
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected vertical bar

v_ceil_f32 v1, |i1+1|
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected vertical bar

//===----------------------------------------------------------------------===//
// Constant expressions may be used with 'abs' and 'neg' modifiers.
//===----------------------------------------------------------------------===//

v_ceil_f32 v1, abs(i1)
// VI: v_ceil_f32_e32 v1, 1            ; encoding: [0x81,0x3a,0x02,0x7e]

v_ceil_f32 v1, abs(i1+1)
// VI: v_ceil_f32_e32 v1, 2            ; encoding: [0x82,0x3a,0x02,0x7e]

v_ceil_f32 v1, abs(-(i1+1))
// VI: v_ceil_f32_e32 v1, 0x7ffffffe   ; encoding: [0xff,0x3a,0x02,0x7e,0xfe,0xff,0xff,0x7f]

v_ceil_f32 v1, abs(fm1)
// VI: v_ceil_f32_e32 v1, 1.0          ; encoding: [0xf2,0x3a,0x02,0x7e]

v_mad_f16 v5, v1, v2, abs(i1)
// VI: v_mad_f16 v5, v1, v2, |1|       ; encoding: [0x05,0x04,0xea,0xd1,0x01,0x05,0x06,0x02]

v_mad_f16 v5, v1, v2, abs(i1+1)
// VI: v_mad_f16 v5, v1, v2, |2|       ; encoding: [0x05,0x04,0xea,0xd1,0x01,0x05,0x0a,0x02]

v_mad_f16 v5, v1, v2, abs(hm1)
// VI: v_mad_f16 v5, v1, v2, |-1.0|    ; encoding: [0x05,0x04,0xea,0xd1,0x01,0x05,0xce,0x03]

v_ceil_f32 v1, neg(i1+1)
// VI: v_ceil_f32_e32 v1, 0x80000002   ; encoding: [0xff,0x3a,0x02,0x7e,0x02,0x00,0x00,0x80]

v_ceil_f32 v1, neg(1+i1)
// VI: v_ceil_f32_e32 v1, 0x80000002   ; encoding: [0xff,0x3a,0x02,0x7e,0x02,0x00,0x00,0x80]

v_ceil_f32 v1, neg(-i1+3)
// VI: v_ceil_f32_e32 v1, 0x80000002   ; encoding: [0xff,0x3a,0x02,0x7e,0x02,0x00,0x00,0x80]

v_ceil_f32 v1, neg(-(i1+1))
// VI: v_ceil_f32_e32 v1, 0x7ffffffe   ; encoding: [0xff,0x3a,0x02,0x7e,0xfe,0xff,0xff,0x7f]

v_ceil_f32 v1, neg(fm1)
// VI: v_ceil_f32_e32 v1, 1.0          ; encoding: [0xf2,0x3a,0x02,0x7e]

v_mad_f16 v5, v1, v2, neg(hm1)
// VI: v_mad_f16 v5, v1, v2, neg(-1.0) ; encoding: [0x05,0x00,0xea,0xd1,0x01,0x05,0xce,0x83]

//===----------------------------------------------------------------------===//
// Constant expressions may be used where inline constants are accepted.
//===----------------------------------------------------------------------===//

v_ceil_f64 v[0:1], i1+1
// VI: v_ceil_f64_e32 v[0:1], 2        ; encoding: [0x82,0x30,0x00,0x7e]

v_and_b32 v0, i1+1, v0
// VI: v_and_b32_e32 v0, 2, v0         ; encoding: [0x82,0x00,0x00,0x26]

v_and_b32 v0, 1+i1, v0
// VI: v_and_b32_e32 v0, 2, v0         ; encoding: [0x82,0x00,0x00,0x26]

v_and_b32 v0, -i1+3, v0
// VI: v_and_b32_e32 v0, 2, v0         ; encoding: [0x82,0x00,0x00,0x26]

v_and_b32 v0, -(i1+1), v0
// VI: v_and_b32_e32 v0, -2, v0        ; encoding: [0xc2,0x00,0x00,0x26]

v_add_u16 v0, (i1+4)/2, v1
// VI: v_add_u16_e32 v0, 2, v1         ; encoding: [0x82,0x02,0x00,0x4c]

buffer_atomic_inc v1, off, s[8:11], i1+1 glc
// VI: buffer_atomic_inc v1, off, s[8:11], 2 glc ; encoding: [0x00,0x40,0x2c,0xe1,0x00,0x01,0x02,0x82]

s_addk_i32 s2, i1+1
// VI: s_addk_i32 s2, 0x2              ; encoding: [0x02,0x00,0x02,0xb7]

s_cmpk_eq_i32 s2, i1+1
// VI: s_cmpk_eq_i32 s2, 0x2           ; encoding: [0x02,0x00,0x02,0xb1]

s_setreg_imm32_b32 0x6, i1+1
// VI: s_setreg_imm32_b32 hwreg(HW_REG_LDS_ALLOC, 0, 1), 2 ; encoding: [0x06,0x00,0x00,0xba,0x02,0x00,0x00,0x00]

v_madak_f16 v1, v2, v3, i1+1
// VI: v_madak_f16 v1, v2, v3, 0x2     ; encoding: [0x02,0x07,0x02,0x4a,0x02,0x00,0x00,0x00]

s_set_gpr_idx_on s0, i1+1
// VI: s_set_gpr_idx_on s0, gpr_idx(SRC1) ; encoding: [0x00,0x02,0x11,0xbf]

s_atc_probe i1-1, s[4:5], i1+1
// VI: s_atc_probe 0, s[4:5], 0x2      ; encoding: [0x02,0x00,0x9a,0xc0,0x02,0x00,0x00,0x00]

s_load_dword s1, s[2:3], i1+1 glc
// VI: s_load_dword s1, s[2:3], 0x2 glc ; encoding: [0x41,0x00,0x03,0xc0,0x02,0x00,0x00,0x00]

//===----------------------------------------------------------------------===//
// Constant expressions may be used where literals are accepted.
//===----------------------------------------------------------------------===//

v_ceil_f64 v[0:1], i1+100
// VI: v_ceil_f64_e32 v[0:1], 0x65     ; encoding: [0xff,0x30,0x00,0x7e,0x65,0x00,0x00,0x00]

v_and_b32 v0, i1+100, v0
// VI: v_and_b32_e32 v0, 0x65, v0      ; encoding: [0xff,0x00,0x00,0x26,0x65,0x00,0x00,0x00]

v_and_b32 v0, -i1+102, v0
// VI: v_and_b32_e32 v0, 0x65, v0      ; encoding: [0xff,0x00,0x00,0x26,0x65,0x00,0x00,0x00]

v_add_u16 v0, (i1+100)*2, v0
// VI: v_add_u16_e32 v0, 0xca, v0      ; encoding: [0xff,0x00,0x00,0x4c,0xca,0x00,0x00,0x00]

//===----------------------------------------------------------------------===//
// Constant expressions may be used with Name:Value modifiers.
//===----------------------------------------------------------------------===//

buffer_load_dword v1, off, s[4:7], s1 offset:-1+1
// VI: buffer_load_dword v1, off, s[4:7], s1 ; encoding: [0x00,0x00,0x50,0xe0,0x00,0x01,0x01,0x01]

buffer_load_dword v1, off, s[4:7], s1 offset:i1+4
// VI: buffer_load_dword v1, off, s[4:7], s1 offset:5 ; encoding: [0x05,0x00,0x50,0xe0,0x00,0x01,0x01,0x01]

buffer_load_dword v1, off, s[4:7], s1 offset:4+i1
// VI: buffer_load_dword v1, off, s[4:7], s1 offset:5 ; encoding: [0x05,0x00,0x50,0xe0,0x00,0x01,0x01,0x01]

buffer_load_dword v1, off, s[4:7], s1 offset:-i1+4
// VI: buffer_load_dword v1, off, s[4:7], s1 offset:3 ; encoding: [0x03,0x00,0x50,0xe0,0x00,0x01,0x01,0x01]

//===----------------------------------------------------------------------===//
// Relocatable expressions can be used with 32-bit instructions.
//===----------------------------------------------------------------------===//

v_ceil_f32 v1, -u
// VI: v_ceil_f32_e32 v1, -u           ; encoding: [0xff,0x3a,0x02,0x7e,A,A,A,A]
// VI-NEXT: ; fixup A - offset: 4, value: -u, kind: FK_PCRel_4

v_and_b32 v0, u+1, v0
// VI: v_and_b32_e32 v0, u+1, v0       ; encoding: [0xff,0x00,0x00,0x26,A,A,A,A]
// VI-NEXT: ;   fixup A - offset: 4, value: u+1, kind: FK_PCRel_4

//===----------------------------------------------------------------------===//
// Relocatable expressions cannot be used as 16/20/21/64-bit operands.
//===----------------------------------------------------------------------===//

v_ceil_f64 v[0:1], u
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_add_u16 v0, u, v0
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_addk_i32 s2, u
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

s_load_dword s1, s[2:3], u glc
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

//===----------------------------------------------------------------------===//
// Relocatable expressions cannot be used with VOP3 modifiers.
//===----------------------------------------------------------------------===//

v_ceil_f32 v1, |u|
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected an absolute expression

v_ceil_f32 v1, neg(u)
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected an absolute expression

v_ceil_f32 v1, abs(u)
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: expected an absolute expression

//===----------------------------------------------------------------------===//
// Misc tests with symbols.
//===----------------------------------------------------------------------===//

.globl global
.globl gds

// Parse a global expression
s_mov_b32 s0, global
// VI: s_mov_b32 s0, global ; encoding: [0xff,0x00,0x80,0xbe,A,A,A,A]
// VI-NEXT: ;   fixup A - offset: 4, value: global, kind: FK_PCRel_4

// Use a token with the same name as a global
ds_gws_init v2 gds
// VI: ds_gws_init v2 gds ; encoding: [0x00,0x00,0x33,0xd9,0x02,0x00,0x00,0x00]

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
// VI: s_mov_b32 s0, 0x202 ; encoding: [0xff,0x00,0x80,0xbe,0x02,0x02,0x00,0x00]

v_mul_f32 v0, foo+2, v2
// VI: v_mul_f32_e32 v0, 0x202, v2 ; encoding: [0xff,0x04,0x00,0x0a,0x02,0x02,0x00,0x00]

BB1:
v_nop_e64
BB2:
s_sub_u32 vcc_lo, vcc_lo, (BB2+4)-BB1
// VI: s_sub_u32 vcc_lo, vcc_lo, (BB2+4)-BB1 ; encoding: [0x6a,0xff,0xea,0x80,A,A,A,A]
// VI-NEXT: ;   fixup A - offset: 4, value: (BB2+4)-BB1, kind: FK_Data_4
s_add_u32 vcc_lo, vcc_lo, (BB2-BB1)&4294967295
// VI: s_add_u32 vcc_lo, vcc_lo, (BB2-BB1)&4294967295 ; encoding: [0x6a,0xff,0x6a,0x80,A,A,A,A]
// VI-NEXT: ;   fixup A - offset: 4, value: (BB2-BB1)&4294967295, kind: FK_Data_4
s_addc_u32 vcc_hi, vcc_hi, (BB2-BB1)>>32
// VI: s_addc_u32 vcc_hi, vcc_hi, (BB2-BB1)>>32 ; encoding: [0x6b,0xff,0x6b,0x82,A,A,A,A]
// VI-NEXT: ;   fixup A - offset: 4, value: (BB2-BB1)>>32, kind: FK_Data_4

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

//===----------------------------------------------------------------------===//
// Symbols may look like registers.
// They should be allowed in expressions if there is no ambiguity.
//===----------------------------------------------------------------------===//

v=1
v_sin_f32 v0, -v
// VI: v_sin_f32_e32 v0, -1            ; encoding: [0xc1,0x52,0x00,0x7e]

s=1
s_not_b32 s0, -s
// VI: s_not_b32 s0, -1                ; encoding: [0xc1,0x04,0x80,0xbe]

ttmp=1
s_not_b32 s0, -ttmp
// VI: s_not_b32 s0, -1                ; encoding: [0xc1,0x04,0x80,0xbe]

//===----------------------------------------------------------------------===//
// Registers have priority over symbols.
//===----------------------------------------------------------------------===//

v=1
v_sin_f32 v0, -v[0]
// VI: v_sin_f32_e64 v0, -v0           ; encoding: [0x00,0x00,0x69,0xd1,0x00,0x01,0x00,0x20]

s0=1
v_sin_f32 v0, -s0
// VI: v_sin_f32_e64 v0, -s0           ; encoding: [0x00,0x00,0x69,0xd1,0x00,0x00,0x00,0x20]

ttmp0=1
v_sin_f32 v0, -[ttmp0]
// VI: v_sin_f32_e64 v0, -ttmp0        ; encoding: [0x00,0x00,0x69,0xd1,0x70,0x00,0x00,0x20]

//===----------------------------------------------------------------------===//
// Incorrect register names and unsupported registers should not be interpreted
// as expressions, rather they should trigger errors.
//===----------------------------------------------------------------------===//

s1000=1
v_sin_f32 v0, -s1000
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

xnack_mask_lo=1
v_sin_f32 v0, xnack_mask_lo
// NOVI: :[[@LINE-1]]:{{[0-9]+}}: error: register not available on this GPU
