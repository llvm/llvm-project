// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefix=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

ds_add_f64 v1, v[2:3] offset:65535
// GFX1210: ds_add_f64 v1, v[2:3] offset:65535 ; encoding: [0xff,0xff,0x50,0xd9,0x01,0x02,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_f64 v1, v[2:3] offset:65535
// GFX12-ERR-NEXT:{{^}}^

ds_add_f64 v255, v[2:3] offset:65535
// GFX1210: ds_add_f64 v255, v[2:3] offset:65535 ; encoding: [0xff,0xff,0x50,0xd9,0xff,0x02,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_f64 v255, v[2:3] offset:65535
// GFX12-ERR-NEXT:{{^}}^

ds_add_f64 v1, v[254:255] offset:65535
// GFX1210: ds_add_f64 v1, v[254:255] offset:65535 ; encoding: [0xff,0xff,0x50,0xd9,0x01,0xfe,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_f64 v1, v[254:255] offset:65535
// GFX12-ERR-NEXT:{{^}}^

ds_add_f64 v1, v[2:3]
// GFX1210: ds_add_f64 v1, v[2:3] ; encoding: [0x00,0x00,0x50,0xd9,0x01,0x02,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_f64 v1, v[2:3]
// GFX12-ERR-NEXT:{{^}}^

ds_add_f64 v1, v[2:3]
// GFX1210: ds_add_f64 v1, v[2:3] ; encoding: [0x00,0x00,0x50,0xd9,0x01,0x02,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_f64 v1, v[2:3]
// GFX12-ERR-NEXT:{{^}}^

ds_add_f64 v1, v[2:3] offset:4
// GFX1210: ds_add_f64 v1, v[2:3] offset:4 ; encoding: [0x04,0x00,0x50,0xd9,0x01,0x02,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_f64 v1, v[2:3] offset:4
// GFX12-ERR-NEXT:{{^}}^

ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:65535
// GFX1210: ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:65535 ; encoding: [0xff,0xff,0xd0,0xd9,0x01,0x02,0x00,0x04]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:65535
// GFX12-ERR-NEXT:{{^}}^

ds_add_rtn_f64 v[254:255], v1, v[2:3] offset:65535
// GFX1210: ds_add_rtn_f64 v[254:255], v1, v[2:3] offset:65535 ; encoding: [0xff,0xff,0xd0,0xd9,0x01,0x02,0x00,0xfe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_rtn_f64 v[254:255], v1, v[2:3] offset:65535
// GFX12-ERR-NEXT:{{^}}^

ds_add_rtn_f64 v[4:5], v255, v[2:3] offset:65535
// GFX1210: ds_add_rtn_f64 v[4:5], v255, v[2:3] offset:65535 ; encoding: [0xff,0xff,0xd0,0xd9,0xff,0x02,0x00,0x04]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_rtn_f64 v[4:5], v255, v[2:3] offset:65535
// GFX12-ERR-NEXT:{{^}}^

ds_add_rtn_f64 v[4:5], v1, v[254:255] offset:65535
// GFX1210: ds_add_rtn_f64 v[4:5], v1, v[254:255] offset:65535 ; encoding: [0xff,0xff,0xd0,0xd9,0x01,0xfe,0x00,0x04]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_rtn_f64 v[4:5], v1, v[254:255] offset:65535
// GFX12-ERR-NEXT:{{^}}^

ds_add_rtn_f64 v[4:5], v1, v[2:3]
// GFX1210: ds_add_rtn_f64 v[4:5], v1, v[2:3] ; encoding: [0x00,0x00,0xd0,0xd9,0x01,0x02,0x00,0x04]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_rtn_f64 v[4:5], v1, v[2:3]
// GFX12-ERR-NEXT:{{^}}^

ds_add_rtn_f64 v[4:5], v1, v[2:3]
// GFX1210: ds_add_rtn_f64 v[4:5], v1, v[2:3] ; encoding: [0x00,0x00,0xd0,0xd9,0x01,0x02,0x00,0x04]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_rtn_f64 v[4:5], v1, v[2:3]
// GFX12-ERR-NEXT:{{^}}^

ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:4
// GFX1210: ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:4 ; encoding: [0x04,0x00,0xd0,0xd9,0x01,0x02,0x00,0x04]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX12-ERR-NEXT: ds_add_rtn_f64 v[4:5], v1, v[2:3] offset:4
// GFX12-ERR-NEXT:{{^}}^
