// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 %s 2>&1 | FileCheck %s -check-prefix=GFX13 --implicit-check-not=error: --strict-whitespace

//===----------------------------------------------------------------------===//
// A VOPD instruction can use only one literal.
//===----------------------------------------------------------------------===//

v_dual_mul_f32      v11, 0x24681357, v2          ::  v_dual_mul_f32      v10, 0xbabe, v5
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_mul_f32      v11, 0x24681357, v2          ::  v_dual_mul_f32      v10, 0xbabe, v5
// GFX13-NEXT:{{^}}                                                                              ^

//===----------------------------------------------------------------------===//
// When 2 different literals are specified, show the location
// of the last literal which is not a KImm, if any.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, v74, 0xa0172923, v161  ::  v_dual_lshlrev_b32  v247, 0xbabe, v99
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_fmamk_f32    v122, v74, 0xa0172923, v161  ::  v_dual_lshlrev_b32  v247, 0xbabe, v99
// GFX13-NEXT:{{^}}                                                                               ^

v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, v3, v1, 0xbabe
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, v3, v1, 0xbabe
// GFX13-NEXT:{{^}}                        ^

v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, 0xbabe, v1, 0xbabe
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, 0xbabe, v1, 0xbabe
// GFX13-NEXT:{{^}}                                                                              ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX13-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX13-NEXT:{{^}}                          ^

//===----------------------------------------------------------------------===//
// Check that assembler detects a different literal regardless of its location.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX13-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0x1234, 0xdeadbeef, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0x1234, 0xdeadbeef, v162
// GFX13-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0x1234, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0x1234, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX13-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0x1234, 0xdeadbeef, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_fmamk_f32    v122, 0x1234, 0xdeadbeef, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX13-NEXT:{{^}}                                                                                   ^

//===----------------------------------------------------------------------===//
// When 2 different literals are specified and all literals are KImm,
// show the location of the last KImm literal.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, s0, 0xdeadbeef, v161   ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX13-NEXT:{{^}}v_dual_fmamk_f32    v122, s0, 0xdeadbeef, v161   ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX13-NEXT:{{^}}                                                                                 ^

//===----------------------------------------------------------------------===//
// A VOPD instruction cannot use more than 2 scalar operands
//===----------------------------------------------------------------------===//

// 2 different SGPRs + LITERAL

v_dual_fmaak_f32    v122, s74, v161, 2.741       ::  v_dual_max_i32       v247, s75, v98
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_fmaak_f32    v122, s74, v161, 2.741       ::  v_dual_max_i32       v247, s75, v98
// GFX13-NEXT:{{^}}                                                                                ^

v_dual_mov_b32      v247, s73                    ::  v_dual_fmaak_f32     v122, s74, v161, 2.741
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_mov_b32      v247, s73                    ::  v_dual_fmaak_f32     v122, s74, v161, 2.741
// GFX13-NEXT:{{^}}                                                                                ^

v_dual_fmamk_f32    v122, s0, 0xbabe, v161       ::  v_dual_fmamk_f32     v123, s1, 0xbabe, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_fmamk_f32    v122, s0, 0xbabe, v161       ::  v_dual_fmamk_f32     v123, s1, 0xbabe, v162
// GFX13-NEXT:{{^}}                                                                                ^

// 2 different SGPRs + VCC

v_dual_add_f32      v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_add_f32      v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX13-NEXT:{{^}}                                                                              ^

v_dual_cndmask_b32   v6, s1, v3                  ::  v_dual_add_f32       v255, s2, v2
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_cndmask_b32   v6, s1, v3                  ::  v_dual_add_f32       v255, s2, v2
// GFX13-NEXT:{{^}}                                                                                ^

v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX13-NEXT:{{^}}                                                                              ^

v_dual_cndmask_b32 v1, s2, v3, vcc_lo :: v_dual_cndmask_b32 v2, s3, v4, vcc_lo
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_cndmask_b32 v1, s2, v3, vcc_lo :: v_dual_cndmask_b32 v2, s3, v4, vcc_lo
// GFX13-NEXT:{{^}}                                                                        ^

// SGPR + LITERAL + VCC

v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_mov_b32       v254, 0xbabe
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_mov_b32       v254, 0xbabe
// GFX13-NEXT:{{^}}                                                                                ^

v_dual_cndmask_b32  v255, 0xbabe, v2             ::  v_dual_mov_b32       v254, s1
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_cndmask_b32  v255, 0xbabe, v2             ::  v_dual_mov_b32       v254, s1
// GFX13-NEXT:{{^}}                                                                                ^

v_dual_cndmask_b32  v255, s3, v2                 ::  v_dual_fmamk_f32     v254, v1, 0xbabe, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_cndmask_b32  v255, s3, v2                 ::  v_dual_fmamk_f32     v254, v1, 0xbabe, v162
// GFX13-NEXT:{{^}}                          ^

v_dual_cndmask_b32  v255, v1, v2                 ::  v_dual_fmamk_f32     v254, s3, 0xbabe, v162
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_cndmask_b32  v255, v1, v2                 ::  v_dual_fmamk_f32     v254, s3, 0xbabe, v162
// GFX13-NEXT:{{^}}                                                                                ^

// SGPR + VCC + VCC_LO
// This is a special case because implicit VCC operand has 64 bit size.
// SP3 does not accept this instruction as well.

v_dual_add_f32      v255, vcc_lo, v2             ::  v_dual_cndmask_b32   v6, s1, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX13-NEXT:{{^}}v_dual_add_f32      v255, vcc_lo, v2             ::  v_dual_cndmask_b32   v6, s1, v3
// GFX13-NEXT:{{^}}                                                                              ^

// FIXME: Error should be 'unsupported instruction'
v_dual_add_f32 v255, v4, v2 :: v_dual_and_b32 v6, v1, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// GFX13-NEXT:{{^}}v_dual_add_f32 v255, v4, v2 :: v_dual_and_b32 v6, v1, v3
// GFX13-NEXT:{{^}}^

v_dual_cndmask_b32 v255, v4, v2 :: v_dual_fmaak_f32 v7, v101, v3, 0xaf123456
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: one dst register must be even and the other odd
// GFX13-NEXT:{{^}}v_dual_cndmask_b32 v255, v4, v2 :: v_dual_fmaak_f32 v7, v101, v3, 0xaf123456
// GFX13-NEXT:{{^}}                                                    ^

v_dual_add_f32 v2, v2, v5 :: v_dual_mul_f32 v4, 130, v6
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: one dst register must be even and the other odd
// GFX13-NEXT:{{^}}v_dual_add_f32 v2, v2, v5 :: v_dual_mul_f32 v4, 130, v6
// GFX13-NEXT:{{^}}                                            ^

// Even though it could be represented as VOPD3, fmac reads its dst and bank constraints still apply to src2.
v_dual_fmac_f32 v255, s105, v2 :: v_dual_fmac_f32 v7, s1, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error:

// Destination should be distinct even if not checked for parity in VOPD3
v_dual_fmac_f32 v7, v4, v2 :: v_dual_fmac_f32 v7, v1, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: dst registers must be distinct
// GFX13-NEXT:{{^}}v_dual_fmac_f32 v7, v4, v2 :: v_dual_fmac_f32 v7, v1, v3
// GFX13-NEXT:{{^}}                                              ^

v_dual_add_f32 v7, v4, v2 :: v_dual_add_f32 v7, v5, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: dst registers must be distinct
// GFX13-NEXT:{{^}}v_dual_add_f32 v7, v4, v2 :: v_dual_add_f32 v7, v5, v3
// GFX13-NEXT:{{^}}                                            ^

//===----------------------------------------------------------------------===//
// Literals not supported by VOPD3. Inline literals can only be encoded for
// src0, but not for vsrc1 or vsrc2.
//===----------------------------------------------------------------------===//
v_dual_fma_f32 v255, s105, v2, v255 :: v_dual_fma_f32 v7, 1, 0, v8
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_dual_fma_f32 v255, s105, v2, v255 :: v_dual_fma_f32 v7, 1, 0, v8
// GFX13-NEXT:{{^}}                                                             ^

v_dual_fma_f32 v255, s105, v2, v255 :: v_dual_fma_f32 v7, 1, v0, 0
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-NEXT:{{^}}v_dual_fma_f32 v255, s105, v2, v255 :: v_dual_fma_f32 v7, 1, v0, 0
// GFX13-NEXT:{{^}}                                                                 ^

//===----------------------------------------------------------------------===//
// Check that we properly detect bank conflicts if instruction is derived from
// VOP3.
//===----------------------------------------------------------------------===//
v_dual_fma_f32 v1, v4, v2, v3 :: v_dual_fma_f32 v3, v8, v7, v6
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error:

v_dual_fma_f32 v1, v4, v2, v3 :: v_dual_fma_f32 v3, v5, v6, v8
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error:

v_dual_fma_f32 v1, v4, v2, v3 :: v_dual_fma_f32 v3, v5, v8, v7
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error:

v_dual_fma_f32 v1, v4, v2, v3 :: v_dual_fmac_f32 v7, v5, v8
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error:

v_dual_fmac_f32 v7, v5, v8 :: v_dual_fma_f32 v1, v4, v2, v3
// GFX13: :[[@LINE-1]]:{{[0-9]+}}: error:

