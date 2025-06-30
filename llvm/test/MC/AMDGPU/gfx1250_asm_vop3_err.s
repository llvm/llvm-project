// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX125X-ERR,GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX125X-ERR-NEXT:{{^}}v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                                          ^

v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] quad_perm:[3,2,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX125X-ERR-NEXT:{{^}}v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] quad_perm:[3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                                          ^
