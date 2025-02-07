// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: -strict-whitespace %s

s_barrier_leave 0
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

s_delay_alu instid0(XDL_DEP_1)
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid value name XDL_DEP_1
