// RUN: llvm-mc -triple=amdgcn -mcpu=gfx950 -show-encoding %s | FileCheck --check-prefix=GFX950 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck --check-prefix=GFX940-ERR --implicit-check-not=error: %s

ds_read_b64_tr_b4 v[0:1], v1
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x00,0x00,0xc0,0xd9,0x01,0x00,0x00,0x00]

ds_read_b64_tr_b4 v[2:3], v3 offset:64
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x40,0x00,0xc0,0xd9,0x03,0x00,0x00,0x02]

ds_read_b64_tr_b8 v[0:1], v1
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x00,0x00,0xc4,0xd9,0x01,0x00,0x00,0x00]

ds_read_b64_tr_b8 v[2:3], v3 offset:64
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x40,0x00,0xc4,0xd9,0x03,0x00,0x00,0x02]

ds_read_b64_tr_b16 v[0:1], v1
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x00,0x00,0xc6,0xd9,0x01,0x00,0x00,0x00]

ds_read_b64_tr_b16 v[2:3], v3 offset:64
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x40,0x00,0xc6,0xd9,0x03,0x00,0x00,0x02]

ds_read_b96_tr_b6 v[0:2], v0
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x00,0x00,0xc2,0xd9,0x00,0x00,0x00,0x00]

ds_read_b96_tr_b6 v[2:4], v2 offset:64
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x40,0x00,0xc2,0xd9,0x02,0x00,0x00,0x02]

ds_read_b96_tr_b6 v[1:3], v0
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x00,0x00,0xc2,0xd9,0x00,0x00,0x00,0x01]

ds_read_b96_tr_b6 v[1:3], v2 offset:64
// GFX940-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX950: encoding: [0x40,0x00,0xc2,0xd9,0x02,0x00,0x00,0x01]
