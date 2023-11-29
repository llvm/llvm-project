// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefix=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=WAVESIZE-ERR --implicit-check-not=error: --strict-whitespace %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23]
// GFX1210: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0x22,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 1.0
// GFX1210: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 1.0 ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 1.0 neg_lo:[0,0,1] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[1,0,0]
// GFX1210: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[1,0,0] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0x22,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[0,1,0]
// GFX1210: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[0,1,0] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0x22,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[0,0,1]
// GFX1210: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[0,0,1] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0x22,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_hi:[0,0,1]
// GFX1210: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_hi:[0,0,1] ; encoding: [0x08,0x44,0x5b,0xcc,0x00,0x09,0x22,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31]
// GFX1210: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[1,0,0]
// GFX1210: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[1,0,0] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0x42,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[0,1,0]
// GFX1210: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[0,1,0] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0x42,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[0,0,1]
// GFX1210: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_hi:[0,0,1]
// GFX1210: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x5c,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11]
// GFX1210: v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11] ; encoding: [0x04,0x40,0x5d,0xcc,0x00,0x05,0x12,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], 1.0
// GFX1210: v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], 1.0 ; encoding: [0x04,0x40,0x5d,0xcc,0x00,0x05,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], 1.0 neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x5d,0xcc,0x00,0x05,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0]
// GFX1210: v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[1,0,0] ; encoding: [0x04,0x40,0x5d,0xcc,0x00,0x05,0x12,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0]
// GFX1210: v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,1,0] ; encoding: [0x04,0x40,0x5d,0xcc,0x00,0x05,0x12,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11] neg_lo:[0,0,1] ; encoding: [0x04,0x40,0x5d,0xcc,0x00,0x05,0x12,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1]
// GFX1210: v_wmma_f32_16x16x4_f32 v[4:11], v[0:1], v[2:3], v[4:11] neg_hi:[0,0,1] ; encoding: [0x04,0x44,0x5d,0xcc,0x00,0x05,0x12,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x16_xf32 v[16:23], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_f32_16x16x16_xf32 v[16:23], v[0:7], v[8:15], v[16:23] ; encoding: [0x10,0x40,0x5e,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x16_xf32 v[16:23], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f32_16x16x16_xf32 v[16:23], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x5e,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23] ; encoding: [0x10,0x40,0x62,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x62,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x62,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x10,0x41,0x62,0xcc,0x00,0x11,0x42,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x10,0x42,0x62,0xcc,0x00,0x11,0x42,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x62,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1]
// GFX1210: v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x62,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] ; encoding: [0x10,0x40,0x60,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x60,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x60,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x10,0x41,0x60,0xcc,0x00,0x11,0x42,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x10,0x42,0x60,0xcc,0x00,0x11,0x42,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x60,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1]
// GFX1210: v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x60,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19]
// GFX1210: v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19] ; encoding: [0x10,0x40,0x61,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x61,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x61,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x10,0x41,0x61,0xcc,0x00,0x11,0x42,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x10,0x42,0x61,0xcc,0x00,0x11,0x42,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x61,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1]
// GFX1210: v_wmma_f16_16x16x32_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x61,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19]
// GFX1210: v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19] ; encoding: [0x10,0x40,0x63,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x63,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x63,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x10,0x41,0x63,0xcc,0x00,0x11,0x42,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x10,0x42,0x63,0xcc,0x00,0x11,0x42,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1]
// GFX1210: v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x63,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1]
// GFX1210: v_wmma_bf16_16x16x32_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x63,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23] ; encoding: [0x1a,0x40,0x64,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], 1.0 ; encoding: [0x1a,0x40,0x64,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x1a,0x40,0x64,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x1a,0x41,0x64,0xcc,0x00,0x11,0x42,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x1a,0x42,0x64,0xcc,0x00,0x11,0x42,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1]
// GFX1210: v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1] ; encoding: [0x1a,0x40,0x64,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1]
// GFX1210: v_wmma_bf16f32_16x16x32_bf16 v[26:29], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1] ; encoding: [0x1a,0x44,0x64,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] ; encoding: [0x10,0x40,0x6a,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x6a,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6a,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6a,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x6a,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] ; encoding: [0x10,0x40,0x6b,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x6b,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6b,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6b,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_fp8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x6b,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] ; encoding: [0x10,0x40,0x6c,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x6c,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6c,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6c,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_bf8_fp8 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x6c,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] ; encoding: [0x10,0x40,0x6d,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x6d,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6d,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6d,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1]
// GFX1210: v_wmma_f32_16x16x64_bf8_bf8 v[16:23], v[0:7], v[8:15], v[16:23] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x6d,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], v[16:19]
// GFX1210: v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] ; encoding: [0x10,0x40,0x6e,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x6e,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6e,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6e,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_fp8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x6e,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], v[16:19]
// GFX1210: v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] ; encoding: [0x10,0x40,0x6f,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x6f,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6f,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x6f,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_fp8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x6f,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], v[16:19]
// GFX1210: v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] ; encoding: [0x10,0x40,0x70,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x70,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x70,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x70,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_bf8_fp8 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x70,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], v[16:19]
// GFX1210: v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] ; encoding: [0x10,0x40,0x71,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], 1.0
// GFX1210: v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x71,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x71,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x71,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1]
// GFX1210: v_wmma_f16_16x16x64_bf8_bf8 v[16:19], v[0:7], v[8:15], v[16:19] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x71,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23] ; encoding: [0x10,0x40,0x72,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], 1
// GFX1210: v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], 1 ; encoding: [0x10,0x40,0x72,0xcc,0x00,0x11,0x06,0x1a]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0]
// GFX1210: v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] ; encoding: [0x10,0x40,0x72,0xcc,0x00,0x11,0x42,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0]
// GFX1210: v_wmma_i32_16x16x64_iu8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] ; encoding: [0x10,0x40,0x72,0xcc,0x00,0x11,0x42,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x128_iu4 v[16:23], v[0:7], v[8:15], v[16:23]
// GFX1210: v_wmma_i32_16x16x128_iu4 v[16:23], v[0:7], v[8:15], v[16:23] ; encoding: [0x10,0x40,0x7c,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x128_iu4 v[16:23], v[0:7], v[8:15], 1
// GFX1210: v_wmma_i32_16x16x128_iu4 v[16:23], v[0:7], v[8:15], 1 ; encoding: [0x10,0x40,0x7c,0xcc,0x00,0x11,0x06,0x1a]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x128_iu4 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0]
// GFX1210: v_wmma_i32_16x16x128_iu4 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] ; encoding: [0x10,0x40,0x7c,0xcc,0x00,0x11,0x42,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_i32_16x16x128_iu4 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0]
// GFX1210: v_wmma_i32_16x16x128_iu4 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] ; encoding: [0x10,0x40,0x7c,0xcc,0x00,0x11,0x42,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32
// GFX1210: v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 ; encoding: [0x18,0x40,0x65,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 index_key:1
// GFX1210: v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 index_key:1 ; encoding: [0x18,0x48,0x65,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x18,0x41,0x65,0xcc,0x00,0x11,0x82,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_swmmac_f32_16x16x64_f16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x18,0x42,0x65,0xcc,0x00,0x11,0x82,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32
// GFX1210: v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 ; encoding: [0x18,0x40,0x66,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 index_key:1
// GFX1210: v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 index_key:1 ; encoding: [0x18,0x48,0x66,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x18,0x41,0x66,0xcc,0x00,0x11,0x82,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_swmmac_f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x18,0x42,0x66,0xcc,0x00,0x11,0x82,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28
// GFX1210: v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 ; encoding: [0x18,0x40,0x67,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 index_key:1
// GFX1210: v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 index_key:1 ; encoding: [0x18,0x48,0x67,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x18,0x41,0x67,0xcc,0x00,0x11,0x72,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_swmmac_f16_16x16x64_f16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x18,0x42,0x67,0xcc,0x00,0x11,0x72,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28
// GFX1210: v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28 ; encoding: [0x18,0x40,0x68,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28 index_key:1
// GFX1210: v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28 index_key:1 ; encoding: [0x18,0x48,0x68,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x18,0x41,0x68,0xcc,0x00,0x11,0x72,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_swmmac_bf16_16x16x64_bf16 v[24:27], v[0:7], v[8:23], v28 neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x18,0x42,0x68,0xcc,0x00,0x11,0x72,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32
// GFX1210: v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 ; encoding: [0x18,0x40,0x69,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 index_key:1
// GFX1210: v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 index_key:1 ; encoding: [0x18,0x48,0x69,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0] neg_hi:[1,0,0]
// GFX1210: v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0] neg_hi:[1,0,0] ; encoding: [0x18,0x41,0x69,0xcc,0x00,0x11,0x82,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0] neg_hi:[0,1,0]
// GFX1210: v_swmmac_bf16f32_16x16x64_bf16 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0] neg_hi:[0,1,0] ; encoding: [0x18,0x42,0x69,0xcc,0x00,0x11,0x82,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], v32
// GFX1210: v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], v32 ; encoding: [0x18,0x40,0x73,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], v32 index_key:1
// GFX1210: v_swmmac_f32_16x16x128_fp8_fp8 v[24:31], v[0:7], v[8:23], v32 index_key:1 ; encoding: [0x18,0x48,0x73,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], v[8:23], v32
// GFX1210: v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], v[8:23], v32 ; encoding: [0x18,0x40,0x74,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], v[8:23], v32 index_key:1
// GFX1210: v_swmmac_f32_16x16x128_fp8_bf8 v[24:31], v[0:7], v[8:23], v32 index_key:1 ; encoding: [0x18,0x48,0x74,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], v[8:23], v32
// GFX1210: v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], v[8:23], v32 ; encoding: [0x18,0x40,0x75,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], v[8:23], v32 index_key:1
// GFX1210: v_swmmac_f32_16x16x128_bf8_fp8 v[24:31], v[0:7], v[8:23], v32 index_key:1 ; encoding: [0x18,0x48,0x75,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], v[8:23], v32
// GFX1210: v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], v[8:23], v32 ; encoding: [0x18,0x40,0x76,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], v[8:23], v32 index_key:1
// GFX1210: v_swmmac_f32_16x16x128_bf8_bf8 v[24:31], v[0:7], v[8:23], v32 index_key:1 ; encoding: [0x18,0x48,0x76,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], v[8:23], v28
// GFX1210: v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], v[8:23], v28 ; encoding: [0x18,0x40,0x77,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], v[8:23], v28 index_key:1
// GFX1210: v_swmmac_f16_16x16x128_fp8_fp8 v[24:27], v[0:7], v[8:23], v28 index_key:1 ; encoding: [0x18,0x48,0x77,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], v[8:23], v28
// GFX1210: v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], v[8:23], v28 ; encoding: [0x18,0x40,0x78,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], v[8:23], v28 index_key:1
// GFX1210: v_swmmac_f16_16x16x128_fp8_bf8 v[24:27], v[0:7], v[8:23], v28 index_key:1 ; encoding: [0x18,0x48,0x78,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], v[8:23], v28
// GFX1210: v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], v[8:23], v28 ; encoding: [0x18,0x40,0x79,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], v[8:23], v28 index_key:1
// GFX1210: v_swmmac_f16_16x16x128_bf8_fp8 v[24:27], v[0:7], v[8:23], v28 index_key:1 ; encoding: [0x18,0x48,0x79,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], v[8:23], v28
// GFX1210: v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], v[8:23], v28 ; encoding: [0x18,0x40,0x7a,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], v[8:23], v28 index_key:1
// GFX1210: v_swmmac_f16_16x16x128_bf8_bf8 v[24:27], v[0:7], v[8:23], v28 index_key:1 ; encoding: [0x18,0x48,0x7a,0xcc,0x00,0x11,0x72,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v32
// GFX1210: v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v32 ; encoding: [0x18,0x40,0x7b,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v32 index_key:1
// GFX1210: v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v32 index_key:1 ; encoding: [0x18,0x48,0x7b,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0]
// GFX1210: v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0] ; encoding: [0x18,0x40,0x7b,0xcc,0x00,0x11,0x82,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0]
// GFX1210: v_swmmac_i32_16x16x128_iu8 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0] ; encoding: [0x18,0x40,0x7b,0xcc,0x00,0x11,0x82,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_i32_16x16x256_iu4 v[24:31], v[0:7], v[8:23], v32
// GFX1210: v_swmmac_i32_16x16x256_iu4 v[24:31], v[0:7], v[8:23], v32 ; encoding: [0x18,0x40,0x7d,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_i32_16x16x256_iu4 v[24:31], v[0:7], v[8:23], v32 index_key:1
// GFX1210: v_swmmac_i32_16x16x256_iu4 v[24:31], v[0:7], v[8:23], v32 index_key:1 ; encoding: [0x18,0x48,0x7d,0xcc,0x00,0x11,0x82,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_i32_16x16x256_iu4 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0]
// GFX1210: v_swmmac_i32_16x16x256_iu4 v[24:31], v[0:7], v[8:23], v32 neg_lo:[1,0,0] ; encoding: [0x18,0x40,0x7d,0xcc,0x00,0x11,0x82,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swmmac_i32_16x16x256_iu4 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0]
// GFX1210: v_swmmac_i32_16x16x256_iu4 v[24:31], v[0:7], v[8:23], v32 neg_lo:[0,1,0] ; encoding: [0x18,0x40,0x7d,0xcc,0x00,0x11,0x82,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX12-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU
