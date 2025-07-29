// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX125X-ERR,GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX125X-ERR-NEXT:{{^}}v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                                          ^

v_mad_u32 v2, v4, v7, v8 dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX125X-ERR-NEXT:{{^}}v_mad_u32 v2, v4, v7, v8 dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                         ^

v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX125X-ERR-NEXT:{{^}}v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                                        ^

v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX125X-ERR-NEXT:{{^}}v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] dpp8:[7,6,5,4,3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                                        ^

v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] quad_perm:[3,2,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX125X-ERR-NEXT:{{^}}v_lshl_add_u64 v[2:3], v[4:5], v7, v[8:9] quad_perm:[3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                                          ^

v_mad_u32 v2, v4, v7, v8 quad_perm:[3,2,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX125X-ERR-NEXT:{{^}}v_mad_u32 v2, v4, v7, v8 quad_perm:[3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                         ^

v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX125X-ERR-NEXT:{{^}}v_mad_nc_u64_u32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                                        ^

v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1251-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: DP ALU dpp only supports row_share
// GFX125X-ERR-NEXT:{{^}}v_mad_nc_i64_i32 v[4:5], v2, v5, v[6:7] quad_perm:[3,2,1,0]
// GFX125X-ERR-NEXT:{{^}}                                        ^
