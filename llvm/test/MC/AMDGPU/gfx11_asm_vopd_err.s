// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck %s -check-prefix=GFX11 --implicit-check-not=error: --strict-whitespace

//===----------------------------------------------------------------------===//
// A VOPD instruction can use only one literal.
//===----------------------------------------------------------------------===//

v_dual_mul_f32      v11, 0x24681357, v2          ::  v_dual_mul_f32      v10, 0xbabe, v5
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_mul_f32      v11, 0x24681357, v2          ::  v_dual_mul_f32      v10, 0xbabe, v5
// GFX11-NEXT:{{^}}                                                                              ^

//===----------------------------------------------------------------------===//
// When 2 different literals are specified, show the location
// of the last literal which is not a KImm, if any.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, v74, 0xa0172923, v161  ::  v_dual_lshlrev_b32  v247, 0xbabe, v99
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, v74, 0xa0172923, v161  ::  v_dual_lshlrev_b32  v247, 0xbabe, v99
// GFX11-NEXT:{{^}}                                                                               ^

v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, v3, v1, 0xbabe
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, v3, v1, 0xbabe
// GFX11-NEXT:{{^}}                        ^

v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, 0xbabe, v1, 0xbabe
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, 0xbabe, v1, 0xbabe
// GFX11-NEXT:{{^}}                                                                              ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX11-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX11-NEXT:{{^}}                          ^

//===----------------------------------------------------------------------===//
// Check that assembler detects a different literal regardless of its location.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX11-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0x1234, 0xdeadbeef, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0x1234, 0xdeadbeef, v162
// GFX11-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0x1234, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0x1234, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX11-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0x1234, 0xdeadbeef, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0x1234, 0xdeadbeef, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX11-NEXT:{{^}}                                                                                   ^

//===----------------------------------------------------------------------===//
// When 2 different literals are specified and all literals are KImm,
// show the location of the last KImm literal.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, s0, 0xdeadbeef, v161   ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, s0, 0xdeadbeef, v161   ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX11-NEXT:{{^}}                                                                                 ^

//===----------------------------------------------------------------------===//
// A VOPD instruction cannot use more than 2 scalar operands.
//===----------------------------------------------------------------------===//

// 2 different SGPRs + LITERAL

v_dual_fmaak_f32    v122, s74, v161, 2.741       ::  v_dual_and_b32       v247, s75, v98
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_fmaak_f32    v122, s74, v161, 2.741       ::  v_dual_and_b32       v247, s75, v98
// GFX11-NEXT:{{^}}                                                                                ^

v_dual_mov_b32      v247, s73                    ::  v_dual_fmaak_f32     v122, s74, v161, 2.741
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_mov_b32      v247, s73                    ::  v_dual_fmaak_f32     v122, s74, v161, 2.741
// GFX11-NEXT:{{^}}                                                                                ^

v_dual_fmamk_f32    v122, s0, 0xbabe, v161       ::  v_dual_fmamk_f32     v123, s1, 0xbabe, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, s0, 0xbabe, v161       ::  v_dual_fmamk_f32     v123, s1, 0xbabe, v162
// GFX11-NEXT:{{^}}                                                                                ^

// 2 different SGPRs + VCC

v_dual_add_f32      v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_add_f32      v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX11-NEXT:{{^}}                                                                              ^

v_dual_cndmask_b32   v6, s1, v3                  ::  v_dual_add_f32       v255, s2, v2
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_cndmask_b32   v6, s1, v3                  ::  v_dual_add_f32       v255, s2, v2
// GFX11-NEXT:{{^}}                                                                                ^

v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX11-NEXT:{{^}}                                                                              ^

// SGPR + LITERAL + VCC

v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_mov_b32       v254, 0xbabe
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_mov_b32       v254, 0xbabe
// GFX11-NEXT:{{^}}                                                                                ^

v_dual_cndmask_b32  v255, 0xbabe, v2             ::  v_dual_mov_b32       v254, s1
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_cndmask_b32  v255, 0xbabe, v2             ::  v_dual_mov_b32       v254, s1
// GFX11-NEXT:{{^}}                                                                                ^

v_dual_cndmask_b32  v255, s3, v2                 ::  v_dual_fmamk_f32     v254, v1, 0xbabe, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_cndmask_b32  v255, s3, v2                 ::  v_dual_fmamk_f32     v254, v1, 0xbabe, v162
// GFX11-NEXT:{{^}}                          ^

v_dual_cndmask_b32  v255, v1, v2                 ::  v_dual_fmamk_f32     v254, s3, 0xbabe, v162
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_cndmask_b32  v255, v1, v2                 ::  v_dual_fmamk_f32     v254, s3, 0xbabe, v162
// GFX11-NEXT:{{^}}                                                                                ^

// SGPR + VCC + VCC_LO
// This is a special case because implicit VCC operand has 64 bit size.
// SP3 does not accept this instruction as well.

v_dual_add_f32      v255, vcc_lo, v2             ::  v_dual_cndmask_b32   v6, s1, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX11-NEXT:{{^}}v_dual_add_f32      v255, vcc_lo, v2             ::  v_dual_cndmask_b32   v6, s1, v3
// GFX11-NEXT:{{^}}                                                                              ^

//===----------------------------------------------------------------------===//
// One dst register must be even and the other odd.
//===----------------------------------------------------------------------===//

v_dual_add_f32      v0, v4, v2                   ::  v_dual_add_f32       v2, v1, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: one dst register must be even and the other odd
// GFX11-NEXT:{{^}}v_dual_add_f32      v0, v4, v2                   ::  v_dual_add_f32       v2, v1, v3
// GFX11-NEXT:{{^}}                                                                          ^

v_dual_mov_b32      v1, v4                       ::  v_dual_add_f32       v5, v1, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: one dst register must be even and the other odd
// GFX11-NEXT:{{^}}v_dual_mov_b32      v1, v4                       ::  v_dual_add_f32       v5, v1, v3
// GFX11-NEXT:{{^}}                                                                          ^

v_dual_cndmask_b32  v2, v4, v5                   ::  v_dual_add_f32       v8, v5, v6
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: one dst register must be even and the other odd
// GFX11-NEXT:{{^}}v_dual_cndmask_b32  v2, v4, v5                   ::  v_dual_add_f32       v8, v5, v6
// GFX11-NEXT:{{^}}                                                                          ^

v_dual_fmac_f32     v3, v4, v5                   ::  v_dual_add_f32       v9, v5, v6
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: one dst register must be even and the other odd
// GFX11-NEXT:{{^}}v_dual_fmac_f32     v3, v4, v5                   ::  v_dual_add_f32       v9, v5, v6
// GFX11-NEXT:{{^}}                                                                          ^

v_dual_fmaak_f32    v4, v4, v5, 0xaf123456       ::  v_dual_add_f32       v0, v5, v6
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: one dst register must be even and the other odd
// GFX11-NEXT:{{^}}v_dual_fmaak_f32    v4, v4, v5, 0xaf123456       ::  v_dual_add_f32       v0, v5, v6
// GFX11-NEXT:{{^}}                                                                          ^

v_dual_fmamk_f32    v5, v4, 0xaf123456, v6       ::  v_dual_add_f32       v1, v5, v6
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: one dst register must be even and the other odd
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v5, v4, 0xaf123456, v6       ::  v_dual_add_f32       v1, v5, v6
// GFX11-NEXT:{{^}}                                                                          ^

//===----------------------------------------------------------------------===//
// Src0 operands must use different VGPR banks.
//===----------------------------------------------------------------------===//

v_dual_add_f32      v1, v1, v5                   ::  v_dual_mov_b32       v2, v1
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_add_f32      v1, v1, v5                   ::  v_dual_mov_b32       v2, v1
// GFX11-NEXT:{{^}}                                                                              ^

v_dual_mov_b32      v1, v2                       ::  v_dual_add_f32       v2, v6, v6
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_mov_b32      v1, v2                       ::  v_dual_add_f32       v2, v6, v6
// GFX11-NEXT:{{^}}                                                                              ^

v_dual_cndmask_b32  v1, v3, v5                   ::  v_dual_add_f32       v2, v11, v6
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_cndmask_b32  v1, v3, v5                   ::  v_dual_add_f32       v2, v11, v6
// GFX11-NEXT:{{^}}                                                                              ^

v_dual_fmac_f32     v1, v4, v5                   ::  v_dual_add_f32       v2, v44, v6
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_fmac_f32     v1, v4, v5                   ::  v_dual_add_f32       v2, v44, v6
// GFX11-NEXT:{{^}}                                                                              ^

v_dual_fmaak_f32    v1, v5, v5, 0xaf123456       ::  v_dual_add_f32       v2, v25, v6
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_fmaak_f32    v1, v5, v5, 0xaf123456       ::  v_dual_add_f32       v2, v25, v6
// GFX11-NEXT:{{^}}                                                                              ^

v_dual_fmamk_f32    v1, v6, 0xaf123456, v6       ::  v_dual_add_f32       v2, v2, v6
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v1, v6, 0xaf123456, v6       ::  v_dual_add_f32       v2, v2, v6
// GFX11-NEXT:{{^}}                                                                              ^

//===----------------------------------------------------------------------===//
// Src1 operands must use different VGPR banks.
//===----------------------------------------------------------------------===//

v_dual_add_f32      v1, v4, v0                   ::  v_dual_add_f32       v2, v5, v4
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src1 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_add_f32      v1, v4, v0                   ::  v_dual_add_f32       v2, v5, v4
// GFX11-NEXT:{{^}}                                                                                  ^

v_dual_cndmask_b32  v1, v4, v1                   ::  v_dual_add_f32       v2, v5, v9
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src1 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_cndmask_b32  v1, v4, v1                   ::  v_dual_add_f32       v2, v5, v9
// GFX11-NEXT:{{^}}                                                                                  ^

v_dual_fmac_f32     v1, v4, v2                   ::  v_dual_add_f32       v2, v5, v14
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src1 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_fmac_f32     v1, v4, v2                   ::  v_dual_add_f32       v2, v5, v14
// GFX11-NEXT:{{^}}                                                                                  ^

v_dual_fmaak_f32    v1, v4, v3, 0xaf123456       ::  v_dual_add_f32       v2, v5, v23
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src1 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_fmaak_f32    v1, v4, v3, 0xaf123456       ::  v_dual_add_f32       v2, v5, v23
// GFX11-NEXT:{{^}}                                                                                  ^

v_dual_add_f32      v2, v4, v4                   ::  v_dual_cndmask_b32   v1, v5, v0
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src1 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_add_f32      v2, v4, v4                   ::  v_dual_cndmask_b32   v1, v5, v0
// GFX11-NEXT:{{^}}                                                                                  ^

v_dual_add_f32      v2, v4, v5                   ::  v_dual_fmac_f32      v1, v5, v1
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src1 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_add_f32      v2, v4, v5                   ::  v_dual_fmac_f32      v1, v5, v1
// GFX11-NEXT:{{^}}                                                                                  ^

v_dual_fmaak_f32    v1, v4, v3, 0xaf123456       ::  v_dual_fmaak_f32     v2, v5, v23, 0xaf123456
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src1 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_fmaak_f32    v1, v4, v3, 0xaf123456       ::  v_dual_fmaak_f32     v2, v5, v23, 0xaf123456
// GFX11-NEXT:{{^}}                                                                                  ^

//===----------------------------------------------------------------------===//
// Src2 operands must use different VGPR banks.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v6, v1, 0xaf123456, v3       :: v_dual_fmamk_f32      v5, v2, 0xaf123456, v5
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src2 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v6, v1, 0xaf123456, v3       :: v_dual_fmamk_f32      v5, v2, 0xaf123456, v5
// GFX11-NEXT:{{^}}                                                                                              ^

v_dual_fmac_f32     v7, v1, v2                   :: v_dual_fmamk_f32      v6, v2, 0xaf123456, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src2 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_fmac_f32     v7, v1, v2                   :: v_dual_fmamk_f32      v6, v2, 0xaf123456, v3
// GFX11-NEXT:{{^}}                                                                                              ^

v_dual_fmamk_f32    v6, v1, 0xaf123456, v3       :: v_dual_fmac_f32       v5, v2, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src2 operands must use different VGPR banks
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v6, v1, 0xaf123456, v3       :: v_dual_fmac_f32       v5, v2, v3
// GFX11-NEXT:{{^}}                                        ^

//===----------------------------------------------------------------------===//
// Check invalid VOPD syntax.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v6, v1, 0xaf123456, v2       : : v_dual_fmac_f32       v5, v2, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: unknown token in expression
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v6, v1, 0xaf123456, v2       : : v_dual_fmac_f32       v5, v2, v3
// GFX11-NEXT:{{^}}                                                 ^

v_dual_fmamk_f32    v6, v1, 0xaf123456, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v6, v1, 0xaf123456, v3
// GFX11-NEXT:{{^}}^

v_dual_fmamk_f32    v6, v1, 0xaf123456, v2       :: v_dual_fmac_f32
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v6, v1, 0xaf123456, v2       :: v_dual_fmac_f32
// GFX11-NEXT:{{^}}^

v_dual_add_f32      v255, v4 :: v_add_f32 v6, v1, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX11-NEXT:{{^}}v_dual_add_f32      v255, v4 :: v_add_f32 v6, v1, v3
// GFX11-NEXT:{{^}}                             ^

v_dual_fmamk_f32    v6, v1, 0xaf123456, v3 ::
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: expected a VOPDY instruction after ::
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v6, v1, 0xaf123456, v3 ::
// GFX11-NEXT:{{^}}                                             ^

v_add_f32           v6, v1, v3 ::
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: expected a VOPDY instruction after ::
// GFX11-NEXT:{{^}}v_add_f32           v6, v1, v3 ::
// GFX11-NEXT:{{^}}                                 ^

v_dual_add_f32      v255::
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: expected a VOPDY instruction after ::
// GFX11-NEXT:{{^}}v_dual_add_f32      v255::
// GFX11-NEXT:{{^}}                          ^

v_dual_add_f32      v255, v4, v2 :: v_add_f32 v6, v1, v3
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: invalid VOPDY instruction
// GFX11-NEXT:{{^}}v_dual_add_f32      v255, v4, v2 :: v_add_f32 v6, v1, v3
// GFX11-NEXT:{{^}}                                    ^
