// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck %s -check-prefix=GFX12 --implicit-check-not=error: --strict-whitespace
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 %s 2>&1 | FileCheck %s -check-prefix=GFX12 --implicit-check-not=error: --strict-whitespace

//===----------------------------------------------------------------------===//
// A VOPD instruction can use only one literal.
//===----------------------------------------------------------------------===//

v_dual_mul_f32      v11, 0x24681357, v2          ::  v_dual_mul_f32      v10, 0xbabe, v5
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_mul_f32      v11, 0x24681357, v2          ::  v_dual_mul_f32      v10, 0xbabe, v5
// GFX12-NEXT:{{^}}                                                                              ^

//===----------------------------------------------------------------------===//
// When 2 different literals are specified, show the location
// of the last literal which is not a KImm, if any.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, v74, 0xa0172923, v161  ::  v_dual_lshlrev_b32  v247, 0xbabe, v99
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_fmamk_f32    v122, v74, 0xa0172923, v161  ::  v_dual_lshlrev_b32  v247, 0xbabe, v99
// GFX12-NEXT:{{^}}                                                                               ^

v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, v3, v1, 0xbabe
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, v3, v1, 0xbabe
// GFX12-NEXT:{{^}}                        ^

v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, 0xbabe, v1, 0xbabe
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, 0xbabe, v1, 0xbabe
// GFX12-NEXT:{{^}}                                                                              ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX12-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX12-NEXT:{{^}}                          ^

//===----------------------------------------------------------------------===//
// Check that assembler detects a different literal regardless of its location.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX12-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0x1234, 0xdeadbeef, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0x1234, 0xdeadbeef, v162
// GFX12-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0x1234, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0x1234, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX12-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0x1234, 0xdeadbeef, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_fmamk_f32    v122, 0x1234, 0xdeadbeef, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX12-NEXT:{{^}}                                                                                   ^

//===----------------------------------------------------------------------===//
// When 2 different literals are specified and all literals are KImm,
// show the location of the last KImm literal.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, s0, 0xdeadbeef, v161   ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: only one unique literal operand is allowed
// GFX12-NEXT:{{^}}v_dual_fmamk_f32    v122, s0, 0xdeadbeef, v161   ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX12-NEXT:{{^}}                                                                                 ^

//===----------------------------------------------------------------------===//
// A VOPD instruction cannot use more than 2 scalar operands
//===----------------------------------------------------------------------===//

// 2 different SGPRs + LITERAL

v_dual_fmaak_f32    v122, s74, v161, 2.741       ::  v_dual_and_b32       v247, s75, v98
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_fmaak_f32    v122, s74, v161, 2.741       ::  v_dual_and_b32       v247, s75, v98
// GFX12-NEXT:{{^}}                                                                                ^

v_dual_mov_b32      v247, s73                    ::  v_dual_fmaak_f32     v122, s74, v161, 2.741
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_mov_b32      v247, s73                    ::  v_dual_fmaak_f32     v122, s74, v161, 2.741
// GFX12-NEXT:{{^}}                                                                                ^

v_dual_fmamk_f32    v122, s0, 0xbabe, v161       ::  v_dual_fmamk_f32     v123, s1, 0xbabe, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_fmamk_f32    v122, s0, 0xbabe, v161       ::  v_dual_fmamk_f32     v123, s1, 0xbabe, v162
// GFX12-NEXT:{{^}}                                                                                ^

// 2 different SGPRs + VCC

v_dual_add_f32      v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_add_f32      v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX12-NEXT:{{^}}                                                                              ^

v_dual_cndmask_b32   v6, s1, v3                  ::  v_dual_add_f32       v255, s2, v2
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_cndmask_b32   v6, s1, v3                  ::  v_dual_add_f32       v255, s2, v2
// GFX12-NEXT:{{^}}                                                                                ^

v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_cndmask_b32   v6, s2, v3
// GFX12-NEXT:{{^}}                                                                              ^

// SGPR + LITERAL + VCC

v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_mov_b32       v254, 0xbabe
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_cndmask_b32  v255, s1, v2                 ::  v_dual_mov_b32       v254, 0xbabe
// GFX12-NEXT:{{^}}                                                                                ^

v_dual_cndmask_b32  v255, 0xbabe, v2             ::  v_dual_mov_b32       v254, s1
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_cndmask_b32  v255, 0xbabe, v2             ::  v_dual_mov_b32       v254, s1
// GFX12-NEXT:{{^}}                                                                                ^

v_dual_cndmask_b32  v255, s3, v2                 ::  v_dual_fmamk_f32     v254, v1, 0xbabe, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_cndmask_b32  v255, s3, v2                 ::  v_dual_fmamk_f32     v254, v1, 0xbabe, v162
// GFX12-NEXT:{{^}}                          ^

v_dual_cndmask_b32  v255, v1, v2                 ::  v_dual_fmamk_f32     v254, s3, 0xbabe, v162
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_cndmask_b32  v255, v1, v2                 ::  v_dual_fmamk_f32     v254, s3, 0xbabe, v162
// GFX12-NEXT:{{^}}                                                                                ^

// SGPR + VCC + VCC_LO
// This is a special case because implicit VCC operand has 64 bit size.
// SP3 does not accept this instruction as well.

v_dual_add_f32      v255, vcc_lo, v2             ::  v_dual_cndmask_b32   v6, s1, v3
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
// GFX12-NEXT:{{^}}v_dual_add_f32      v255, vcc_lo, v2             ::  v_dual_cndmask_b32   v6, s1, v3
// GFX12-NEXT:{{^}}                                                                              ^
