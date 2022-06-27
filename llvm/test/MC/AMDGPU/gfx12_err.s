// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: -strict-whitespace %s

s_prefetch_inst s[14:15], 0xffffff, m0, 7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 24-bit signed offset
// GFX12-ERR: s_prefetch_inst s[14:15], 0xffffff, m0, 7
// GFX12-ERR:                           ^
