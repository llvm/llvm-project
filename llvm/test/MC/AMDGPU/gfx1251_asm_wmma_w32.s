// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1251 -show-encoding %s | FileCheck --check-prefix=GFX1251 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1251 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s 2>&1 | FileCheck --check-prefix=WAVESIZE-ERR --implicit-check-not=error: --strict-whitespace %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23]
// GFX1251: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0x22,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 1.0
// GFX1251: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 1.0 ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 1.0 neg_lo:[0,0,1]
// GFX1251: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], 1.0 neg_lo:[0,0,1] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[1,0,0]
// GFX1251: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[1,0,0] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0x22,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[0,1,0]
// GFX1251: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[0,1,0] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0x22,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[0,0,1]
// GFX1251: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_lo:[0,0,1] ; encoding: [0x08,0x40,0x5b,0xcc,0x00,0x09,0x22,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_hi:[0,0,1]
// GFX1251: v_wmma_f64_16x16x4_f64 v[8:23], v[0:3], v[4:7], v[8:23] neg_hi:[0,0,1] ; encoding: [0x08,0x44,0x5b,0xcc,0x00,0x09,0x22,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31]
// GFX1251: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 1.0
// GFX1251: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 1.0 ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0xca,0x1b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1]
// GFX1251: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], 1.0 neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0xca,0x9b]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[1,0,0]
// GFX1251: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[1,0,0] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0x42,0x3c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[0,1,0]
// GFX1251: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[0,1,0] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0x42,0x5c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[0,0,1]
// GFX1251: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_lo:[0,0,1] ; encoding: [0x10,0x40,0x5c,0xcc,0x00,0x11,0x42,0x9c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_hi:[0,0,1]
// GFX1251: v_wmma_f64_16x16x8_f64 v[16:31], v[0:7], v[8:15], v[16:31] neg_hi:[0,0,1] ; encoding: [0x10,0x44,0x5c,0xcc,0x00,0x11,0x42,0x1c]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU
