// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck %s -check-prefix=GFX11 --implicit-check-not=error: --strict-whitespace

s_waitcnt_vscnt s0, 0x1234
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 must be null
// GFX11-NEXT:{{^}}s_waitcnt_vscnt s0, 0x1234
// GFX11-NEXT:{{^}}                ^

s_waitcnt_vmcnt exec_lo, 0x1234
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 must be null
// GFX11-NEXT:{{^}}s_waitcnt_vmcnt exec_lo, 0x1234
// GFX11-NEXT:{{^}}                ^

s_waitcnt_expcnt vcc_lo, 0x1234
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 must be null
// GFX11-NEXT:{{^}}s_waitcnt_expcnt vcc_lo, 0x1234
// GFX11-NEXT:{{^}}                 ^

s_waitcnt_lgkmcnt m0, 0x1234
// GFX11: :[[@LINE-1]]:{{[0-9]+}}: error: src0 must be null
// GFX11-NEXT:{{^}}s_waitcnt_lgkmcnt m0, 0x1234
// GFX11-NEXT:{{^}}                  ^
