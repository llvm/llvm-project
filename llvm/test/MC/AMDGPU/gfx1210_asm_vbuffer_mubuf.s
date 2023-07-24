// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefix=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

buffer_load_b32 v5, v1, s[8:11], s3 offen offset:4095 nv
// GFX1210: buffer_load_b32 v5, v1, s[8:11], s3 offen offset:4095 nv ; encoding: [0x83,0x00,0x05,0xc4,0x05,0x10,0x80,0x40,0x01,0xff,0x0f,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}buffer_load_b32 v5, v1, s[8:11], s3 offen offset:4095 nv
// GFX12-ERR-NEXT:{{^}}                                                      ^

buffer_store_b128 v[1:4], v0, s[12:15], s4 idxen offset:4095 nv
// GFX1210: buffer_store_b128 v[1:4], v0, s[12:15], s4 idxen offset:4095 nv ; encoding: [0x84,0x40,0x07,0xc4,0x01,0x18,0x80,0x80,0x00,0xff,0x0f,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}buffer_store_b128 v[1:4], v0, s[12:15], s4 idxen offset:4095 nv
// GFX12-ERR-NEXT:{{^}}                                                             ^

buffer_atomic_and_b32 v5, v1, s[8:11], s3 offen offset:4095 nv
// GFX1210: buffer_atomic_and_b32 v5, v1, s[8:11], s3 offen offset:4095 nv ; encoding: [0x83,0x00,0x0f,0xc4,0x05,0x10,0x80,0x40,0x01,0xff,0x0f,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}buffer_atomic_and_b32 v5, v1, s[8:11], s3 offen offset:4095 nv
// GFX12-ERR-NEXT:{{^}}                                                            ^
