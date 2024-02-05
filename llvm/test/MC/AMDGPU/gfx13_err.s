// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding < %s 2>&1 | FileCheck -strict-whitespace -implicit-check-not=error: -check-prefix=GFX13-ERR %s

s_waitcnt 0
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
