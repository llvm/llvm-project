; RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

tensor_load_to_lds s[0:3], s[4:11] r128
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must set modifier r128=0

tensor_load_to_lds s[0:3], s[4:11] th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for load instructions

tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] r128
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must set modifier r128=0

tensor_load_to_lds s[0:3], s[4:11], s[12:15], s[16:19] th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for load instructions

tensor_store_from_lds s[0:3], s[4:11] r128
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must set modifier r128=0

tensor_store_from_lds s[0:3], s[4:11] th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for store instructions

tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19] r128
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must set modifier r128=0

tensor_store_from_lds s[0:3], s[4:11], s[12:15], s[16:19] th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for store instructions
