// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefixes=GFX1210 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR %s

ds_atomic_async_barrier_arrive_b64 v1 offset0:127 offset1:255
// GFX1210: ds_atomic_async_barrier_arrive_b64 v1 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x58,0xd9,0x01,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_atomic_async_barrier_arrive_b64 v255 offset0:16 offset1:4
// GFX1210: ds_atomic_async_barrier_arrive_b64 v255 offset0:16 offset1:4 ; encoding: [0x10,0x04,0x58,0xd9,0xff,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

ds_atomic_async_barrier_arrive_b64 v5
// GFX1210: ds_atomic_async_barrier_arrive_b64 v5   ; encoding: [0x00,0x00,0x58,0xd9,0x05,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
