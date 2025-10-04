// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

s_buffer_load_i8 s5, s[4:7], s0 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}s_buffer_load_i8 s5, s[4:7], s0 scale_offset
// GFX1250-ERR-NEXT:{{^}}                                ^

s_prefetch_data s[18:19], 100, s10, 7 nv
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}s_prefetch_data s[18:19], 100, s10, 7 nv
// GFX1250-ERR-NEXT:{{^}}                                      ^

s_prefetch_data s[18:19], 100, s10, 7 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}s_prefetch_data s[18:19], 100, s10, 7 scale_offset
// GFX1250-ERR-NEXT:{{^}}                                      ^
