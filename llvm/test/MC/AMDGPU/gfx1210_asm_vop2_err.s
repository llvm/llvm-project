// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

v_add_nc_u64 v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_add_nc_u64 v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                    ^

v_sub_nc_u64 v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_sub_nc_u64 v[2:3], v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX1210-ERR-NEXT:{{^}}                                    ^
