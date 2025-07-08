// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefixes=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR %s

ds_atomic_async_barrier_arrive_b64 v1 offset:65407
// GFX1250: ds_atomic_async_barrier_arrive_b64 v1 offset:65407 ; encoding: [0x7f,0xff,0x58,0xd9,0x01,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_atomic_async_barrier_arrive_b64 v255 offset:1040
// GFX1250: ds_atomic_async_barrier_arrive_b64 v255 offset:1040 ; encoding: [0x10,0x04,0x58,0xd9,0xff,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_atomic_async_barrier_arrive_b64 v5
// GFX1250: ds_atomic_async_barrier_arrive_b64 v5   ; encoding: [0x00,0x00,0x58,0xd9,0x05,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_atomic_barrier_arrive_rtn_b64 v[2:3], v2, v[4:5]
// GFX1250: ds_atomic_barrier_arrive_rtn_b64 v[2:3], v2, v[4:5] ; encoding: [0x00,0x00,0xd4,0xd9,0x02,0x04,0x00,0x02]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_atomic_barrier_arrive_rtn_b64 v[2:3], v2, v[4:5] offset:513
// GFX1250: ds_atomic_barrier_arrive_rtn_b64 v[2:3], v2, v[4:5] offset:513 ; encoding: [0x01,0x02,0xd4,0xd9,0x02,0x04,0x00,0x02]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_atomic_barrier_arrive_rtn_b64 v[254:255], v2, v[4:5] offset:65535
// GFX1250: ds_atomic_barrier_arrive_rtn_b64 v[254:255], v2, v[4:5] offset:65535 ; encoding: [0xff,0xff,0xd4,0xd9,0x02,0x04,0x00,0xfe]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
