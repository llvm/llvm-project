// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX13-ERR --implicit-check-not=error: -strict-whitespace %s

s_barrier_leave
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

s_wakeup
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
