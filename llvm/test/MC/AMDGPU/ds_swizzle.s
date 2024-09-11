// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx908 -show-encoding %s | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10PLUS %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck -check-prefix=GFX10PLUS %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s | FileCheck -check-prefix=GFX10PLUS %s

// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx908 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR %s --implicit-check-not=error:

//==============================================================================
// FFT mode

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,0)
// CHECK:     [0x00,0xe0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0x00,0xe0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,5)
// CHECK:     [0x05,0xe0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0x05,0xe0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,16)
// CHECK:     [0x10,0xe0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0x10,0xe0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,31)
// CHECK:     [0x1f,0xe0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0x1f,0xe0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,32)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: FFT swizzle must be in the interval [0,31]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,-2)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: FFT swizzle must be in the interval [0,31]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a comma

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,16,31)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a closing parentheses


//==============================================================================
// ROTATE mode

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,0)
// CHECK:     [0x00,0xc0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0x00,0xc0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,0)
// CHECK:     [0x00,0xc4,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0x00,0xc4,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,1)
// CHECK:     [0x20,0xc0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0x20,0xc0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,1)
// CHECK:     [0x20,0xc4,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0x20,0xc4,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,31)
// CHECK:     [0xe0,0xc3,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0xe0,0xc3,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,31)
// CHECK:     [0xe0,0xc7,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: [0xe0,0xc7,0xd4,0xd8,0x01,0x00,0x00,0x05]


ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,2,31)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: direction must be 0 (left) or 1 (right)

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,-1,31)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: direction must be 0 (left) or 1 (right)

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,32)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: number of threads to rotate must be in the interval [0,31]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,-2)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: number of threads to rotate must be in the interval [0,31]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a comma

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a comma

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a comma

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,1,2)
// ERROR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a closing parentheses




