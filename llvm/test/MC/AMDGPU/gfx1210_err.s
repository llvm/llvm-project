// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX1210-ERR --implicit-check-not=error: -strict-whitespace %s

s_set_vgpr_msb -1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: s_set_vgpr_msb accepts values in range [0..15]
// GFX1210-ERR: s_set_vgpr_msb -1
// GFX1210-ERR:                ^

s_set_vgpr_msb 16
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: s_set_vgpr_msb accepts values in range [0..15]
// GFX1210-ERR: s_set_vgpr_msb 16
// GFX1210-ERR:                ^

s_load_b32 s4, s[2:3], 10 th:TH_LOAD_NT th:TH_LOAD_NT
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_load_b32 s4, s[2:3], 10 th:TH_LOAD_NT th:TH_LOAD_NT
// GFX1210-ERR:                                         ^

s_load_b32 s4, s[2:3], 10 scope:SCOPE_SE scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_load_b32 s4, s[2:3], 10 scope:SCOPE_SE scope:SCOPE_SE
// GFX1210-ERR:                                          ^

s_load_b32 s4, s[2:3], 10 nv nv
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR: s_load_b32 s4, s[2:3], 10 nv nv
// GFX1210-ERR:                              ^
