// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize64 -show-encoding %s | FileCheck %s --check-prefix=GFX10
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize64 %s 2>&1 | FileCheck -check-prefix=NOGFX10 --implicit-check-not=error: %s

i1=1

//===----------------------------------------------------------------------===//
// Constant expressions may be used where literals are accepted.
//===----------------------------------------------------------------------===//

v_bfe_u32 v0, i1+100, v1, v2
// GFX10: v_bfe_u32 v0, 0x65, v1, v2      ; encoding: [0x00,0x00,0x48,0xd5,0xff,0x02,0x0a,0x04,0x65,0x00,0x00,0x00]

v_bfe_u32 v0, v1, i1-100, v2
// GFX10: v_bfe_u32 v0, v1, 0xffffff9d, v2 ; encoding: [0x00,0x00,0x48,0xd5,0x01,0xff,0x09,0x04,0x9d,0xff,0xff,0xff]

v_bfe_u32 v0, v1, v2, (i1+100)*2
// GFX10: v_bfe_u32 v0, v1, v2, 0xca      ; encoding: [0x00,0x00,0x48,0xd5,0x01,0x05,0xfe,0x03,0xca,0x00,0x00,0x00]

v_cmp_f_i32 s[10:11], (i1+100)*2, v2
// GFX10: v_cmp_f_i32_e64 s[10:11], 0xca, v2   ; encoding: [0x0a,0x00,0x80,0xd4,0xff,0x04,0x02,0x00,0xca,0x00,0x00,0x00]

v_cmpx_f_i64 v[1:2], i1+100
// GFX10: v_cmpx_f_i64_e64 v[1:2], 0x65   ; encoding: [0x7e,0x00,0xb0,0xd4,0x01,0xff,0x01,0x00,0x65,0x00,0x00,0x00]

v_lshlrev_b64 v[5:6], i1+0xFFE, v[2:3]
// GFX10: v_lshlrev_b64 v[5:6], 0xfff, v[2:3] ; encoding: [0x05,0x00,0xff,0xd6,0xff,0x04,0x02,0x00,0xff,0x0f,0x00,0x00]

//===----------------------------------------------------------------------===//
// Relocatable expressions can be used with 32-bit instructions.
//===----------------------------------------------------------------------===//

v_bfe_u32 v0, u, v1, v2
// GFX10: v_bfe_u32 v0, u, v1, v2         ; encoding: [0x00,0x00,0x48,0xd5,0xff,0x02,0x0a,0x04,A,A,A,A]
// GFX10-NEXT:                            ;   fixup A - offset: 8, value: u, kind: FK_PCRel_4

v_bfe_u32 v0, v1, u-1, v2
// GFX10: v_bfe_u32 v0, v1, u-1, v2       ; encoding: [0x00,0x00,0x48,0xd5,0x01,0xff,0x09,0x04,A,A,A,A]
// GFX10-NEXT:                            ;   fixup A - offset: 8, value: u-1, kind: FK_Data_4

v_bfe_u32 v0, v1, v2, u+1
// GFX10: v_bfe_u32 v0, v1, v2, u+1       ; encoding: [0x00,0x00,0x48,0xd5,0x01,0x05,0xfe,0x03,A,A,A,A]
// GFX10-NEXT:                            ;   fixup A - offset: 8, value: u+1, kind: FK_PCRel_4

v_cmp_f_i32 s[10:11], u+1, v2
// GFX10: v_cmp_f_i32_e64 s[10:11], u+1, v2 ; encoding: [0x0a,0x00,0x80,0xd4,0xff,0x04,0x02,0x00,A,A,A,A]
// GFX10-NEXT:                              ;   fixup A - offset: 8, value: u+1, kind: FK_PCRel_4

v_lshlrev_b64 v[5:6], u-1, v[2:3]
// GFX10: v_lshlrev_b64 v[5:6], u-1, v[2:3] ; encoding: [0x05,0x00,0xff,0xd6,0xff,0x04,0x02,0x00,A,A,A,A]
// GFX10-NEXT:                              ;   fixup A - offset: 8, value: u-1, kind: FK_Data_4

//===----------------------------------------------------------------------===//
// Instructions can use only one literal.
// Relocatable expressions are counted as literals.
//===----------------------------------------------------------------------===//

s_sub_u32 s0, 123, u
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed

s_sub_u32 s0, u, u
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed

s_sub_u32 s0, u, u1
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed

v_bfe_u32 v0, v2, 123, u
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed

v_bfe_u32 v0, v2, u, u
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed

v_bfe_u32 v0, v2, u, u1
// NOGFX10: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
