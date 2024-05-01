// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX1210-ERR --implicit-check-not=error: -strict-whitespace %s

s_setkill 0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
