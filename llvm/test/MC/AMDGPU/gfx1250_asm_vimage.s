; RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefixes=GFX1250 %s
; RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

tensor_load_to_lds s[0:3], s[4:11]
// GFX1250: tensor_load_to_lds s[0:3], s[4:11] ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x00,0x00,0x00,0x04,0x7c,0x7c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1250: tensor_load_to_lds s[0:3], s[4:11] th:TH_LOAD_BYPASS scope:SCOPE_SYS ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x3c,0x00,0x00,0x04,0x7c,0x7c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19]
// GFX1250: tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x00,0x00,0x00,0x04,0x0c,0x10]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1250: tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] th:TH_LOAD_NT_HT scope:SCOPE_DEV ; encoding: [0x01,0x00,0x71,0xd0,0x00,0x00,0x68,0x00,0x00,0x04,0x0c,0x10]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_store_from_lds s[0:3], s[4:11]
// GFX1250: tensor_store_from_lds s[0:3], s[4:11] ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x00,0x00,0x00,0x04,0x7c,0x7c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_store_from_lds s[0:3], s[4:11] th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1250: tensor_store_from_lds s[0:3], s[4:11] th:TH_STORE_BYPASS scope:SCOPE_SYS ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x3c,0x00,0x00,0x04,0x7c,0x7c]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19]
// GFX1250: tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19] ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x00,0x00,0x00,0x04,0x0c,0x10]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19] th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1250: tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19] th:TH_STORE_NT_HT scope:SCOPE_DEV ; encoding: [0x01,0x40,0x71,0xd0,0x00,0x00,0x68,0x00,0x00,0x04,0x0c,0x10]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
