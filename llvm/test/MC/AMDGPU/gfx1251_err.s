// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1251 -filetype=null %s 2>&1 | FileCheck --check-prefixes=GFX1251-ERR --implicit-check-not=error: -strict-whitespace %s

v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1251-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1251-ERR: v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1251-ERR:                          ^
