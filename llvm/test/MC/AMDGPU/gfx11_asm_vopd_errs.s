// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck %s -check-prefix=GFX11 --implicit-check-not=error: --strict-whitespace

//===----------------------------------------------------------------------===//
// A VOPD instruction can use only one literal.
//===----------------------------------------------------------------------===//

v_dual_mul_f32      v11, 0x24681357, v2          ::  v_dual_mul_f32      v10, 0xbabe, v5
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_mul_f32      v11, 0x24681357, v2          ::  v_dual_mul_f32      v10, 0xbabe, v5
// GFX11-NEXT:{{^}}                                                                              ^

//===----------------------------------------------------------------------===//
// When 2 different literals are specified, show the location
// of the last literal which is not a KImm, if any.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, v74, 0xa0172923, v161  ::  v_dual_lshlrev_b32  v247, 0xbabe, v99
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, v74, 0xa0172923, v161  ::  v_dual_lshlrev_b32  v247, 0xbabe, v99
// GFX11-NEXT:{{^}}                                                                               ^

v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, v3, v1, 0xbabe
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, v3, v1, 0xbabe
// GFX11-NEXT:{{^}}                        ^

v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, 0xbabe, v1, 0xbabe
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_add_f32      v5, 0xaf123456, v2           ::  v_dual_fmaak_f32     v6, 0xbabe, v1, 0xbabe
// GFX11-NEXT:{{^}}                                                                              ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX11-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX11-NEXT:{{^}}                          ^

//===----------------------------------------------------------------------===//
// Check that assembler detects a different literal regardless of its location.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0x1234, v162
// GFX11-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0x1234, 0xdeadbeef, v162
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0xdeadbeef, v161 ::  v_dual_fmamk_f32  v123, 0x1234, 0xdeadbeef, v162
// GFX11-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0xdeadbeef, 0x1234, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0xdeadbeef, 0x1234, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX11-NEXT:{{^}}                                                                                   ^

v_dual_fmamk_f32    v122, 0x1234, 0xdeadbeef, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, 0x1234, 0xdeadbeef, v161     ::  v_dual_fmamk_f32  v123, 0xdeadbeef, 0xdeadbeef, v162
// GFX11-NEXT:{{^}}                                                                                   ^

//===----------------------------------------------------------------------===//
// When 2 different literals are specified and all literals are KImm,
// show the location of the last KImm literal.
//===----------------------------------------------------------------------===//

v_dual_fmamk_f32    v122, s0, 0xdeadbeef, v161   ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX11: error: only one literal operand is allowed
// GFX11-NEXT:{{^}}v_dual_fmamk_f32    v122, s0, 0xdeadbeef, v161   ::  v_dual_fmamk_f32  v123, s0, 0x1234, v162
// GFX11-NEXT:{{^}}                                                                                 ^
