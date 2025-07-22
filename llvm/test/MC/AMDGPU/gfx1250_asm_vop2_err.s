// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_add_f64 v[1:2], v[1:2], v[1:2]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

v_fmaak_f32 v4, v2, v6, 3 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_fmaak_f32 v4, v2, v6, 3 row_share:1
// GFX1250-ERR-NEXT:{{^}}                          ^

v_fmamk_f32 v4, v2, 3, v6 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_fmamk_f32 v4, v2, 3, v6 row_share:1
// GFX1250-ERR-NEXT:{{^}}                          ^

v_fmaak_f16 v4, v2, v6, 3 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_fmaak_f16 v4, v2, v6, 3 row_share:1
// GFX1250-ERR-NEXT:{{^}}                          ^

v_fmamk_f16 v4, v2, 3, v6 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_fmamk_f16 v4, v2, 3, v6 row_share:1
// GFX1250-ERR-NEXT:{{^}}                          ^
