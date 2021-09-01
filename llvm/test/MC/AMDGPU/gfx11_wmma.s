// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -show-encoding %s | FileCheck --check-prefix=W32 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -show-encoding %s | FileCheck --check-prefix=W64 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W32-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=W64-ERR --implicit-check-not=error: %s

//
// Test v_wmma_f32_16x16x16_f16
//

v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23]
// W32: encoding: [0x10,0x40,0x40,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19]
// W64: encoding: [0x10,0x40,0x40,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:23], 1.0, v[8:15], v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:19], 1.0, v[8:15], v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], 1.0, v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:19], v[0:7], 1.0, v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], 1.0
// W32: encoding: [0x10,0x40,0x40,0xcc,0x00,0x11,0xca,0x1b]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:19], v[0:7], v[8:15], 1.0
// W64: encoding: [0x10,0x40,0x40,0xcc,0x00,0x11,0xca,0x1b]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W32: encoding: [0x10,0x41,0x40,0xcc,0x00,0x11,0x42,0x3c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W64: encoding: [0x10,0x41,0x40,0xcc,0x00,0x11,0x42,0x3c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W32: encoding: [0x10,0x42,0x40,0xcc,0x00,0x11,0x42,0x5c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W64: encoding: [0x10,0x42,0x40,0xcc,0x00,0x11,0x42,0x5c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W32: encoding: [0x10,0x43,0x40,0xcc,0x00,0x11,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W64: encoding: [0x10,0x43,0x40,0xcc,0x00,0x11,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] clamp
// W32: encoding: [0x10,0xc0,0x40,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] clamp
// W64: encoding: [0x10,0xc0,0x40,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

//
// Test v_wmma_f32_16x16x16_bf16
//

v_wmma_f32_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
// W32: encoding: [0x10,0x40,0x41,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19]
// W64: encoding: [0x10,0x40,0x41,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:23], 1.0, v[8:15], v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:19], 1.0, v[8:15], v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:23], v[0:7], 1.0, v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:19], v[0:7], 1.0, v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:23], v[0:7], v[8:15], 1.0
// W32: encoding: [0x10,0x40,0x41,0xcc,0x00,0x11,0xca,0x1b]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:19], v[0:7], v[8:15], 1.0
// W64: encoding: [0x10,0x40,0x41,0xcc,0x00,0x11,0xca,0x1b]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W32: encoding: [0x10,0x41,0x41,0xcc,0x00,0x11,0x42,0x3c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W64: encoding: [0x10,0x41,0x41,0xcc,0x00,0x11,0x42,0x3c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W32: encoding: [0x10,0x42,0x41,0xcc,0x00,0x11,0x42,0x5c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W64: encoding: [0x10,0x42,0x41,0xcc,0x00,0x11,0x42,0x5c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W32: encoding: [0x10,0x43,0x41,0xcc,0x00,0x11,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W64: encoding: [0x10,0x43,0x41,0xcc,0x00,0x11,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] clamp
// W32: encoding: [0x10,0xc0,0x41,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f32_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] clamp
// W64: encoding: [0x10,0xc0,0x41,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

//
// Test v_wmma_f16_16x16x16_f16
//

v_wmma_f16_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23]
// W32: encoding: [0x10,0x40,0x42,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19]
// W64: encoding: [0x10,0x40,0x42,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:23], 1.0, v[8:15], v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:19], 1.0, v[8:15], v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:23], v[0:7], 1.0, v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:19], v[0:7], 1.0, v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:23], v[0:7], v[8:15], 1.0
// W32: encoding: [0x10,0x40,0x42,0xcc,0x00,0x11,0xca,0x1b]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:19], v[0:7], v[8:15], 1.0
// W64: encoding: [0x10,0x40,0x42,0xcc,0x00,0x11,0xca,0x1b]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] op_sel:[0,0,1]
// W32: encoding: [0x10,0x60,0x42,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] op_sel:[0,0,1]
// W64: encoding: [0x10,0x60,0x42,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W32: encoding: [0x10,0x41,0x42,0xcc,0x00,0x11,0x42,0x3c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W64: encoding: [0x10,0x41,0x42,0xcc,0x00,0x11,0x42,0x3c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W32: encoding: [0x10,0x42,0x42,0xcc,0x00,0x11,0x42,0x5c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W64: encoding: [0x10,0x42,0x42,0xcc,0x00,0x11,0x42,0x5c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W32: encoding: [0x10,0x43,0x42,0xcc,0x00,0x11,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W64: encoding: [0x10,0x43,0x42,0xcc,0x00,0x11,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23] clamp
// W32: encoding: [0x10,0xc0,0x42,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_f16_16x16x16_f16 v[16:19], v[0:7], v[8:15], v[16:19] clamp
// W64: encoding: [0x10,0xc0,0x42,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

//
// Test v_wmma_bf16_16x16x16_bf16
//

v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
// W32: encoding: [0x10,0x40,0x43,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19]
// W64: encoding: [0x10,0x40,0x43,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:23], 1.0, v[8:15], v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:19], 1.0, v[8:15], v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7], 1.0, v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:19], v[0:7], 1.0, v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7], v[8:15], 1.0
// W32: encoding: [0x10,0x40,0x43,0xcc,0x00,0x11,0xca,0x1b]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:19], v[0:7], v[8:15], 1.0
// W64: encoding: [0x10,0x40,0x43,0xcc,0x00,0x11,0xca,0x1b]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] op_sel:[0,0,1]
// W32: encoding: [0x10,0x60,0x43,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] op_sel:[0,0,1]
// W64: encoding: [0x10,0x60,0x43,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W32: encoding: [0x10,0x41,0x43,0xcc,0x00,0x11,0x42,0x3c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W64: encoding: [0x10,0x41,0x43,0xcc,0x00,0x11,0x42,0x3c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W32: encoding: [0x10,0x42,0x43,0xcc,0x00,0x11,0x42,0x5c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W64: encoding: [0x10,0x42,0x43,0xcc,0x00,0x11,0x42,0x5c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W32: encoding: [0x10,0x43,0x43,0xcc,0x00,0x11,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W64: encoding: [0x10,0x43,0x43,0xcc,0x00,0x11,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23] clamp
// W32: encoding: [0x10,0xc0,0x43,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_bf16_16x16x16_bf16 v[16:19], v[0:7], v[8:15], v[16:19] clamp
// W64: encoding: [0x10,0xc0,0x43,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error


//
// Test v_wmma_i32_16x16x16_iu8
//

v_wmma_i32_16x16x16_iu8 v[16:23], v[0:7], v[8:15], v[16:23]
// W32: encoding: [0x10,0x40,0x44,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:19], v[0:7], v[8:15], v[16:19]
// W64: encoding: [0x10,0x40,0x44,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:23], 1, v[8:15], v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:19], 1, v[8:15], v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:23], v[0:7], 1, v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:19], v[0:7], 1, v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:23], v[0:7], v[8:15], 1
// W32: encoding: [0x10,0x40,0x44,0xcc,0x00,0x11,0x06,0x1a]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:19], v[0:7], v[8:15], 1
// W64: encoding: [0x10,0x40,0x44,0xcc,0x00,0x11,0x06,0x1a]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W32: encoding: [0x10,0x41,0x44,0xcc,0x00,0x11,0x42,0x3c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W64: encoding: [0x10,0x41,0x44,0xcc,0x00,0x11,0x42,0x3c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W32: encoding: [0x10,0x42,0x44,0xcc,0x00,0x11,0x42,0x5c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W64: encoding: [0x10,0x42,0x44,0xcc,0x00,0x11,0x42,0x5c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W32: encoding: [0x10,0x43,0x44,0xcc,0x00,0x11,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W64: encoding: [0x10,0x43,0x44,0xcc,0x00,0x11,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:23], v[0:7], v[8:15], v[16:23] clamp
// W32: encoding: [0x10,0xc0,0x44,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu8 v[16:19], v[0:7], v[8:15], v[16:19] clamp
// W64: encoding: [0x10,0xc0,0x44,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error


//
// Test v_wmma_i32_16x16x16_iu4
//

v_wmma_i32_16x16x16_iu4 v[16:23], v[0:7], v[8:15], v[16:23]
// W32: encoding: [0x10,0x40,0x45,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:19], v[0:7], v[8:15], v[16:19]
// W64: encoding: [0x10,0x40,0x45,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:23], 1.0, v[8:15], v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:19], 1.0, v[8:15], v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:23], v[0:7], 1.0, v[16:23]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:19], v[0:7], 1.0, v[16:19]
// W32-ERR: :[[@LINE-1]]:{{[0-9]+}}: error
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error
v_wmma_i32_16x16x16_iu4 v[16:23], v[0:7], v[8:15], 1
// W32: encoding: [0x10,0x40,0x45,0xcc,0x00,0x11,0x06,0x1a]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:19], v[0:7], v[8:15], 1
// W64: encoding: [0x10,0x40,0x45,0xcc,0x00,0x11,0x06,0x1a]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W32: encoding: [0x10,0x41,0x45,0xcc,0x00,0x11,0x42,0x3c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,0,0] neg_hi:[1,0,0]
// W64: encoding: [0x10,0x41,0x45,0xcc,0x00,0x11,0x42,0x3c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W32: encoding: [0x10,0x42,0x45,0xcc,0x00,0x11,0x42,0x5c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[0,1,0] neg_hi:[0,1,0]
// W64: encoding: [0x10,0x42,0x45,0xcc,0x00,0x11,0x42,0x5c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:23], v[0:7], v[8:15], v[16:23] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W32: encoding: [0x10,0x43,0x45,0xcc,0x00,0x11,0x42,0x7c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:19], v[0:7], v[8:15], v[16:19] neg_lo:[1,1,0] neg_hi:[1,1,0]
// W64: encoding: [0x10,0x43,0x45,0xcc,0x00,0x11,0x42,0x7c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:23], v[0:7], v[8:15], v[16:23] clamp
// W32: encoding: [0x10,0xc0,0x45,0xcc,0x00,0x11,0x42,0x1c]
// W64-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

v_wmma_i32_16x16x16_iu4 v[16:19], v[0:7], v[8:15], v[16:19] clamp
// W64: encoding: [0x10,0xc0,0x45,0xcc,0x00,0x11,0x42,0x1c]
// W32-ERR: :[[@LINE-2]]:{{[0-9]+}}: error

