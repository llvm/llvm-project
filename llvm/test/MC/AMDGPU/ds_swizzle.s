// RUN: not llvm-mc -triple=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck -check-prefix=GFX7 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=GFX8 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx908 -show-encoding %s | FileCheck -check-prefix=GFX9 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10PLUS %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck -check-prefix=GFX10PLUS %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s | FileCheck -check-prefix=GFX10PLUS %s

// RUN: not llvm-mc -triple=amdgcn -mcpu=bonaire -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR-PREGFX9 %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR-PREGFX9 %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx908 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR %s --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERROR %s --implicit-check-not=error:

//==============================================================================
// FFT mode

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,0)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: FFT mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(FFT,0) ; encoding: [0x00,0xe0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(FFT,0) ; encoding: [0x00,0xe0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,5)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: FFT mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(FFT,5) ; encoding: [0x05,0xe0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(FFT,5) ; encoding: [0x05,0xe0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,16)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: FFT mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(FFT,16) ; encoding: [0x10,0xe0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(FFT,16) ; encoding: [0x10,0xe0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,31)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: FFT mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(FFT,31) ; encoding: [0x1f,0xe0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(FFT,31) ; encoding: [0x1f,0xe0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:0xf000
// GFX7:   ds_swizzle_b32 v5, v1 offset:61440 ; encoding: [0x00,0xf0,0xd4,0xd8,0x01,0x00,0x00,0x05]
// GFX8:   ds_swizzle_b32 v5, v1 offset:61440 ; encoding: [0x00,0xf0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(FFT,0) ; encoding: [0x00,0xf0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(FFT,0) ; encoding: [0x00,0xf0,0xd4,0xd8,0x01,0x00,0x00,0x05]


ds_swizzle_b32 v5, v1 offset:swizzle(FFT,32)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: FFT mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: FFT swizzle must be in the interval [0,31]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,-2)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: FFT mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: FFT swizzle must be in the interval [0,31]

ds_swizzle_b32 v5, v1 offset:swizzle(FFT)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: FFT mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: expected a comma

ds_swizzle_b32 v5, v1 offset:swizzle(FFT,16,31)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: FFT mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: expected a closing parentheses

//==============================================================================
// ROTATE mode

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,0)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,0) ; encoding: [0x00,0xc0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,0) ; encoding: [0x00,0xc0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,0)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,0) ; encoding: [0x00,0xc4,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,0) ; encoding: [0x00,0xc4,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,1)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,1) ; encoding: [0x20,0xc0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,1) ; encoding: [0x20,0xc0,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,1)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,1) ; encoding: [0x20,0xc4,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,1) ; encoding: [0x20,0xc4,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,31)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,31) ; encoding: [0xe0,0xc3,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,31) ; encoding: [0xe0,0xc3,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,31)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// GFX9:      ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,31) ; encoding: [0xe0,0xc7,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1,31) ; encoding: [0xe0,0xc7,0xd4,0xd8,0x01,0x00,0x00,0x05]

ds_swizzle_b32 v5, v1 offset:0xd000
// GFX7:   ds_swizzle_b32 v5, v1 offset:53248 ; encoding: [0x00,0xd0,0xd4,0xd8,0x01,0x00,0x00,0x05]
// GFX8:   ds_swizzle_b32 v5, v1 offset:53248 ; encoding: [0x00,0xd0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX9:   ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,0) ; encoding: [0x00,0xd0,0x7a,0xd8,0x01,0x00,0x00,0x05]
// GFX10PLUS: ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,0) ; encoding: [0x00,0xd0,0xd4,0xd8,0x01,0x00,0x00,0x05]


ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,2,31)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: direction must be 0 (left) or 1 (right)

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,-1,31)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: direction must be 0 (left) or 1 (right)

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,32)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: number of threads to rotate must be in the interval [0,31]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,-2)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: number of threads to rotate must be in the interval [0,31]

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: expected a comma

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: expected a comma

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,1)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: expected a comma

ds_swizzle_b32 v5, v1 offset:swizzle(ROTATE,0,1,2)
// ERROR-PREGFX9: :[[@LINE-1]]:{{[0-9]+}}: error: Rotate mode swizzle not supported on this GPU
// ERROR: :[[@LINE-2]]:{{[0-9]+}}: error: expected a closing parentheses
