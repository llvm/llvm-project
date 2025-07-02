// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

tensor_save s[0:1]
// GFX1250: tensor_save s[0:1] ; encoding: [0x00,0x80,0x1b,0xee,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_save s[0:1] th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1250: tensor_save s[0:1] th:TH_STORE_BYPASS scope:SCOPE_SYS ; encoding: [0x00,0x80,0x1b,0xee,0x00,0x00,0x3c,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_save s[0:1] offset:32
// GFX1250: tensor_save s[0:1] offset:32 ; encoding: [0x00,0x80,0x1b,0xee,0x00,0x00,0x00,0x00,0x00,0x20,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_stop
// GFX1250: tensor_stop ; encoding: [0x7c,0xc0,0x1b,0xee,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_stop th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1250: tensor_stop th:TH_STORE_BYPASS scope:SCOPE_SYS ; encoding: [0x7c,0xc0,0x1b,0xee,0x00,0x00,0x3c,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
