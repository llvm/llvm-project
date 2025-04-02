// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx942 %s 2>&1 | FileCheck --check-prefixes=CHECK --implicit-check-not=error: %s

buffer_store_lds_dword s[4:7], -1 offset:4095 lds
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_wbinvl1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_wbinvl1_vol
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

