// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefixes=GFX90A --implicit-check-not=error: %s

v_add_f64 v[1:2], v[1:2], v[1:2]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

global_load_dwordx2 v[1:2], v[0:1], off
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

global_load_dwordx3 v[1:3], v[0:1], off
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

global_load_dwordx4 v[1:4], v[0:1], off
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

global_load_dwordx2 a[1:2], v[0:1], off
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

global_load_dwordx3 a[1:3], v[0:1], off
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

global_load_dwordx4 a[1:4], v[0:1], off
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned


image_load v[1:2], v2, s[0:7] dmask:0x3 unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_load v[1:3], v2, s[0:7] dmask:0x7 unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_load v[1:4], v2, s[0:7] dmask:0xf unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_load a[1:2], v2, s[0:7] dmask:0x3 unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_load a[1:3], v2, s[0:7] dmask:0x7 unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_load a[1:4], v2, s[0:7] dmask:0xf unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode


image_store v[193:194], v[238:241], s[28:35] dmask:0x3 unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_store v[193:195], v[238:241], s[28:35] dmask:0x7 unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_store v[193:196], v[238:241], s[28:35] dmask:0xf unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_store a[193:194], v[238:241], s[28:35] dmask:0x3 unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_store a[193:195], v[238:241], s[28:35] dmask:0x7 unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_store a[193:196], v[238:241], s[28:35] dmask:0xf unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode


image_atomic_swap v4, v[193:196], s[28:35] dmask:0x1 unorm glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_atomic_swap v[5:6], v1, s[8:15] dmask:0x3 unorm
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode


image_atomic_cmpswap v[5:6], v[192:195], s[28:35] dmask:0x3 unorm glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_atomic_cmpswap v[4:5], v[193:196], s[28:35] dmask:0x3 unorm glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_atomic_cmpswap v[5:8], v[192:195], s[28:35] dmask:0xf unorm glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_atomic_cmpswap v[4:7], v[193:196], s[28:35] dmask:0xf unorm glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode


image_atomic_cmpswap a[5:6], v[192:195], s[28:35] dmask:0x3 unorm glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_atomic_cmpswap a[4:5], v[193:196], s[28:35] dmask:0x3 unorm glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_atomic_cmpswap a[5:8], v[192:195], s[28:35] dmask:0xf unorm glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

image_atomic_cmpswap a[4:7], v[193:196], s[28:35] dmask:0xf unorm glc
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode


v_mfma_f32_32x32x8f16 a[0:15], a[1:2], v[0:1], a[0:15]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

v_mfma_i32_4x4x4i8 a[1:4], a0, v1, 2
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

v_mfma_f32_16x16x1f32 a[0:15], a0, v1, a[17:32]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

v_mfma_f32_32x32x1f32 a[0:31], v0, v1, a[33:64]
// GFX90A: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned
