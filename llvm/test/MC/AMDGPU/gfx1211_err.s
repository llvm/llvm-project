// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1211 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX1211-ERR --implicit-check-not=error: -strict-whitespace %s

v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1211-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX1211-ERR: v_mov_b64 v[4:5], v[2:3] quad_perm:[1,1,1,1]
// GFX1211-ERR:                          ^
