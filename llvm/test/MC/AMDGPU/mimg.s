// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI --check-prefix=SICIVI
// RUN: not llvm-mc -triple=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=SICI --check-prefix=SICIVI
// RUN: not llvm-mc -triple=amdgcn -mcpu=fiji -show-encoding %s | FileCheck %s --check-prefix=GCN  --check-prefix=SICIVI --check-prefix=VI --check-prefix=GFX89 --check-prefix=GFX8_0
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx810 -show-encoding %s | FileCheck %s --check-prefix=GCN  --check-prefix=SICIVI --check-prefix=VI --check-prefix=GFX89 --check-prefix=GFX8_1
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=GFX9 --check-prefix=GFX89

// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck %s --check-prefix=NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=fiji %s 2>&1 | FileCheck %s --check-prefix=NOVI --check-prefix=NOGFX8_0 --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx810 %s 2>&1 | FileCheck %s --check-prefix=NOVI --check-prefix=NOGFX8_1 --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --check-prefix=NOGFX9 --implicit-check-not=error:

//===----------------------------------------------------------------------===//
// Image Load/Store
//===----------------------------------------------------------------------===//

image_load    v[4:6], v[237:240], s[28:35] dmask:0x7 unorm
// GCN:  image_load v[4:6], v[237:240], s[28:35] dmask:0x7 unorm ; encoding: [0x00,0x17,0x00,0xf0,0xed,0x04,0x07,0x00]

image_load    v4, v237, s[28:35]
// GCN:  image_load v4, v237, s[28:35] ; encoding: [0x00,0x00,0x00,0xf0,0xed,0x04,0x07,0x00]

image_load    v4, v[237:238], s[28:35]
// GCN:  image_load v4, v[237:238], s[28:35] ; encoding: [0x00,0x00,0x00,0xf0,0xed,0x04,0x07,0x00]

image_load    v4, v[237:239], s[28:35]
// GCN:  image_load v4, v[237:239], s[28:35] ; encoding: [0x00,0x00,0x00,0xf0,0xed,0x04,0x07,0x00]

image_load    v4, v[237:240], s[28:35]
// GCN:  image_load v4, v[237:240], s[28:35] ; encoding: [0x00,0x00,0x00,0xf0,0xed,0x04,0x07,0x00]

image_load    v[4:7], v[237:240], s[28:35] dmask:0x7 tfe
// GCN:  image_load v[4:7], v[237:240], s[28:35] dmask:0x7 tfe ; encoding: [0x00,0x07,0x01,0xf0,0xed,0x04,0x07,0x00]

// Verify support of all possible modifiers.
// FIXME: This test is incorrect because r128 assumes a 128-bit SRSRC.
image_load    v[5:6], v[1:4], s[8:15] dmask:0x1 unorm glc slc r128 tfe lwe da d16
// NOSICI: :[[@LINE-1]]:{{[0-9]+}}: error: d16 modifier is not supported on this GPU
// VI:     image_load v[5:6], v[1:4], s[8:15] dmask:0x1 unorm glc slc r128 tfe lwe da d16 ; encoding: [0x00,0xf1,0x03,0xf2,0x01,0x05,0x02,0x80]
// NOGFX9: :[[@LINE-3]]:{{[0-9]+}}: error: r128 modifier is not supported on this GPU

image_load v5, v[1:4], s[8:15] d16
// NOSICI: :[[@LINE-1]]:{{[0-9]+}}: error: d16 modifier is not supported on this GPU
// GFX89:  image_load v5, v[1:4], s[8:15] d16 ; encoding: [0x00,0x00,0x00,0xf0,0x01,0x05,0x02,0x80]

image_load v5, v[1:4], s[8:15] r128
// SICIVI: image_load v5, v[1:4], s[8:15] r128 ; encoding: [0x00,0x80,0x00,0xf0,0x01,0x05,0x02,0x00]
// NOGFX9: :[[@LINE-2]]:{{[0-9]+}}: error: r128 modifier is not supported on this GPU

image_store   v[193:195], v[237:240], s[28:35] dmask:0x7 unorm
// GCN: image_store v[193:195], v[237:240], s[28:35] dmask:0x7 unorm ; encoding: [0x00,0x17,0x20,0xf0,0xed,0xc1,0x07,0x00]

image_store   v193, v237, s[28:35]
// GCN: image_store v193, v237, s[28:35] ; encoding: [0x00,0x00,0x20,0xf0,0xed,0xc1,0x07,0x00]

image_store   v193, v[237:238], s[28:35]
// GCN: image_store v193, v[237:238], s[28:35] ; encoding: [0x00,0x00,0x20,0xf0,0xed,0xc1,0x07,0x00]

image_store   v193, v[237:239], s[28:35]
// GCN: image_store v193, v[237:239], s[28:35] ; encoding: [0x00,0x00,0x20,0xf0,0xed,0xc1,0x07,0x00]

image_store   v193, v[237:240], s[28:35]
// GCN: image_store v193, v[237:240], s[28:35] ; encoding: [0x00,0x00,0x20,0xf0,0xed,0xc1,0x07,0x00]

image_store   v[193:194], v[237:240], s[28:35] tfe
// GCN: image_store v[193:194], v[237:240], s[28:35] tfe ; encoding: [0x00,0x00,0x21,0xf0,0xed,0xc1,0x07,0x00]

// Verify support of all possible modifiers.
// FIXME: This test is incorrect because r128 assumes a 128-bit SRSRC.
image_store   v5, v[1:4], s[8:15] dmask:0x1 unorm glc slc r128 lwe da d16
// NOSICI: :[[@LINE-1]]:{{[0-9]+}}: error: d16 modifier is not supported on this GPU
// VI:     image_store v5, v[1:4], s[8:15] dmask:0x1 unorm glc slc r128 lwe da d16 ; encoding: [0x00,0xf1,0x22,0xf2,0x01,0x05,0x02,0x80]
// NOGFX9: :[[@LINE-3]]:{{[0-9]+}}: error: r128 modifier is not supported on this GPU

image_store    v5, v[1:4], s[8:15] d16
// NOSICI: :[[@LINE-1]]:{{[0-9]+}}: error: d16 modifier is not supported on this GPU
// GFX89:  image_store v5, v[1:4], s[8:15] d16 ; encoding: [0x00,0x00,0x20,0xf0,0x01,0x05,0x02,0x80]

// FIXME: This test is incorrect because r128 assumes a 128-bit SRSRC.
image_store    v5, v[1:4], s[8:15] r128
// SICIVI: image_store v5, v[1:4], s[8:15] r128 ; encoding: [0x00,0x80,0x20,0xf0,0x01,0x05,0x02,0x00]
// NOGFX9: :[[@LINE-2]]:{{[0-9]+}}: error: r128 modifier is not supported on this GPU

//===----------------------------------------------------------------------===//
// Image Load/Store: d16 unpacked
//===----------------------------------------------------------------------===//

image_load v[5:6], v[1:4], s[8:15] dmask:0x3 d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// GFX8_0:   image_load v[5:6], v[1:4], s[8:15] dmask:0x3 d16 ; encoding: [0x00,0x03,0x00,0xf0,0x01,0x05,0x02,0x80]
// NOGFX8_1: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask, d16 and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe

image_load v[5:7], v[1:4], s[8:15] dmask:0x7 d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// GFX8_0:   image_load v[5:7], v[1:4], s[8:15] dmask:0x7 d16 ; encoding: [0x00,0x07,0x00,0xf0,0x01,0x05,0x02,0x80]
// NOGFX8_1: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask, d16 and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe

image_load v[5:8], v[1:4], s[8:15] dmask:0xf d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// GFX8_0:   image_load v[5:8], v[1:4], s[8:15] dmask:0xf d16 ; encoding: [0x00,0x0f,0x00,0xf0,0x01,0x05,0x02,0x80]
// NOGFX8_1: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask, d16 and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe

image_load v[5:7], v[1:4], s[8:15] dmask:0x3 tfe d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// GFX8_0:   image_load v[5:7], v[1:4], s[8:15] dmask:0x3 tfe d16 ; encoding: [0x00,0x03,0x01,0xf0,0x01,0x05,0x02,0x80]
// NOGFX8_1: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask, d16 and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe

image_load v[5:8], v[1:4], s[8:15] dmask:0x7 tfe d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// GFX8_0:   image_load v[5:8], v[1:4], s[8:15] dmask:0x7 tfe d16 ; encoding: [0x00,0x07,0x01,0xf0,0x01,0x05,0x02,0x80]
// NOGFX8_1: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask, d16 and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe

//===----------------------------------------------------------------------===//
// Image Load/Store: d16 packed
//===----------------------------------------------------------------------===//

image_load v5, v[1:4], s[8:15] dmask:0x3 d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// NOGFX8_0: :[[@LINE-2]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// GFX8_1:   image_load v5, v[1:4], s[8:15] dmask:0x3 d16 ; encoding: [0x00,0x03,0x00,0xf0,0x01,0x05,0x02,0x80]
// GFX9:     image_load v5, v[1:4], s[8:15] dmask:0x3 d16 ; encoding: [0x00,0x03,0x00,0xf0,0x01,0x05,0x02,0x80]

image_load v[5:6], v[1:4], s[8:15] dmask:0x7 d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// NOGFX8_0: :[[@LINE-2]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// GFX8_1:   image_load v[5:6], v[1:4], s[8:15] dmask:0x7 d16 ; encoding: [0x00,0x07,0x00,0xf0,0x01,0x05,0x02,0x80]
// GFX9:     image_load v[5:6], v[1:4], s[8:15] dmask:0x7 d16 ; encoding: [0x00,0x07,0x00,0xf0,0x01,0x05,0x02,0x80]

image_load v[5:6], v[1:4], s[8:15] dmask:0xf d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// NOGFX8_0: :[[@LINE-2]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// GFX8_1:   image_load v[5:6], v[1:4], s[8:15] dmask:0xf d16 ; encoding: [0x00,0x0f,0x00,0xf0,0x01,0x05,0x02,0x80]
// GFX9:     image_load v[5:6], v[1:4], s[8:15] dmask:0xf d16 ; encoding: [0x00,0x0f,0x00,0xf0,0x01,0x05,0x02,0x80]

image_load v[5:6], v[1:4], s[8:15] dmask:0x3 tfe d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// NOGFX8_0: :[[@LINE-2]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// GFX8_1:   image_load v[5:6], v[1:4], s[8:15] dmask:0x3 tfe d16 ; encoding: [0x00,0x03,0x01,0xf0,0x01,0x05,0x02,0x80]
// GFX9:     image_load v[5:6], v[1:4], s[8:15] dmask:0x3 tfe d16 ; encoding: [0x00,0x03,0x01,0xf0,0x01,0x05,0x02,0x80]

image_load v[5:7], v[1:4], s[8:15] dmask:0x7 tfe d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// NOGFX8_0: :[[@LINE-2]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// GFX8_1:   image_load v[5:7], v[1:4], s[8:15] dmask:0x7 tfe d16 ; encoding: [0x00,0x07,0x01,0xf0,0x01,0x05,0x02,0x80]
// GFX9:     image_load v[5:7], v[1:4], s[8:15] dmask:0x7 tfe d16 ; encoding: [0x00,0x07,0x01,0xf0,0x01,0x05,0x02,0x80]

image_load v[5:7], v[1:4], s[8:15] dmask:0xf tfe d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// NOGFX8_0: :[[@LINE-2]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// GFX8_1:   image_load v[5:7], v[1:4], s[8:15] dmask:0xf tfe d16 ; encoding: [0x00,0x0f,0x01,0xf0,0x01,0x05,0x02,0x80]
// GFX9:     image_load v[5:7], v[1:4], s[8:15] dmask:0xf tfe d16 ; encoding: [0x00,0x0f,0x01,0xf0,0x01,0x05,0x02,0x80]

//===----------------------------------------------------------------------===//
// Image Load/Store: a16
//===----------------------------------------------------------------------===//

image_load v5, v[1:2], s[8:15] unorm a16
// GFX9:     image_load v5, v[1:2], s[8:15] unorm a16 ; encoding: [0x00,0x90,0x00,0xf0,0x01,0x05,0x02,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_load v[5:6], v[1:2], s[8:15] dmask:0x3 unorm a16
// GFX9:     image_load v[5:6], v[1:2], s[8:15] dmask:0x3 unorm a16 ; encoding: [0x00,0x93,0x00,0xf0,0x01,0x05,0x02,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_load v[5:7], v[1:2], s[8:15] dmask:0x7 unorm a16
// GFX9:     image_load v[5:7], v[1:2], s[8:15] dmask:0x7 unorm a16 ; encoding: [0x00,0x97,0x00,0xf0,0x01,0x05,0x02,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_load v[5:8], v[1:2], s[8:15] dmask:0xf unorm a16
// GFX9:     image_load v[5:8], v[1:2], s[8:15] dmask:0xf unorm a16 ; encoding: [0x00,0x9f,0x00,0xf0,0x01,0x05,0x02,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_store v5, v[1:2], s[8:15] unorm a16
// GFX9:     image_store v5, v[1:2], s[8:15] unorm a16 ; encoding: [0x00,0x90,0x20,0xf0,0x01,0x05,0x02,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_store v[5:6], v[1:2], s[8:15] dmask:0x3 unorm a16
// GFX9:     image_store v[5:6], v[1:2], s[8:15] dmask:0x3 unorm a16 ; encoding: [0x00,0x93,0x20,0xf0,0x01,0x05,0x02,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_store v[5:7], v[1:2], s[8:15] dmask:0x7 unorm a16
// GFX9:     image_store v[5:7], v[1:2], s[8:15] dmask:0x7 unorm a16 ; encoding: [0x00,0x97,0x20,0xf0,0x01,0x05,0x02,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_store v[5:8], v[1:2], s[8:15] dmask:0xf unorm a16
// GFX9:     image_store v[5:8], v[1:2], s[8:15] dmask:0xf unorm a16 ; encoding: [0x00,0x9f,0x20,0xf0,0x01,0x05,0x02,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

//===----------------------------------------------------------------------===//
// Image Load/Store: a16 & d16
//===----------------------------------------------------------------------===//

image_load v5, v[1:2], s[8:15] dmask:0x3 unorm a16 d16
// GFX9:     image_load v5, v[1:2], s[8:15] dmask:0x3 unorm a16 d16 ; encoding: [0x00,0x93,0x00,0xf0,0x01,0x05,0x02,0x80]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_load v[5:6], v[1:2], s[8:15] dmask:0x7 unorm a16 d16
// GFX9:     image_load v[5:6], v[1:2], s[8:15] dmask:0x7 unorm a16 d16 ; encoding: [0x00,0x97,0x00,0xf0,0x01,0x05,0x02,0x80]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_load v[5:6], v[1:2], s[8:15] dmask:0xf unorm a16 d16
// GFX9:     image_load v[5:6], v[1:2], s[8:15] dmask:0xf unorm a16 d16 ; encoding: [0x00,0x9f,0x00,0xf0,0x01,0x05,0x02,0x80]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_store v5, v[1:2], s[8:15] dmask:0x3 unorm a16 d16
// GFX9:     image_store v5, v[1:2], s[8:15] dmask:0x3 unorm a16 d16 ; encoding: [0x00,0x93,0x20,0xf0,0x01,0x05,0x02,0x80]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_store v[5:6], v[1:2], s[8:15] dmask:0x7 unorm a16 d16
// GFX9:     image_store v[5:6], v[1:2], s[8:15] dmask:0x7 unorm a16 d16 ; encoding: [0x00,0x97,0x20,0xf0,0x01,0x05,0x02,0x80]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

image_store v[5:6], v[1:2], s[8:15] dmask:0xf unorm a16 d16
// GFX9:     image_store v[5:6], v[1:2], s[8:15] dmask:0xf unorm a16 d16 ; encoding: [0x00,0x9f,0x20,0xf0,0x01,0x05,0x02,0x80]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:     error: a16 modifier is not supported on this GPU

//===----------------------------------------------------------------------===//
// Image Load/Store: PCK variants
//===----------------------------------------------------------------------===//

image_load_mip_pck v5, v1, s[8:15] dmask:0x1
// GCN: image_load_mip_pck v5, v1, s[8:15] dmask:0x1 ; encoding: [0x00,0x01,0x10,0xf0,0x01,0x05,0x02,0x00]

image_load_mip_pck v5, v[1:2], s[8:15] dmask:0x1
// GCN: image_load_mip_pck v5, v[1:2], s[8:15] dmask:0x1 ; encoding: [0x00,0x01,0x10,0xf0,0x01,0x05,0x02,0x00]

image_load_mip_pck v5, v[1:3], s[8:15] dmask:0x1
// GCN: image_load_mip_pck v5, v[1:3], s[8:15] dmask:0x1 ; encoding: [0x00,0x01,0x10,0xf0,0x01,0x05,0x02,0x00]

image_load_mip_pck v5, v[1:4], s[8:15] dmask:0x1
// GCN: image_load_mip_pck v5, v[1:4], s[8:15] dmask:0x1 ; encoding: [0x00,0x01,0x10,0xf0,0x01,0x05,0x02,0x00]

image_load_mip_pck v[5:6], v[1:4], s[8:15] dmask:0x3
// GCN: image_load_mip_pck v[5:6], v[1:4], s[8:15] dmask:0x3 ; encoding: [0x00,0x03,0x10,0xf0,0x01,0x05,0x02,0x00]

image_load_mip_pck v[5:6], v[1:4], s[8:15] dmask:0x1 unorm glc slc tfe lwe da
// GCN: image_load_mip_pck v[5:6], v[1:4], s[8:15] dmask:0x1 unorm glc slc tfe lwe da ; encoding: [0x00,0x71,0x13,0xf2,0x01,0x05,0x02,0x00]

image_load_mip_pck_sgn v[5:6], v[1:4], s[8:15] dmask:0x5
// GCN: image_load_mip_pck_sgn v[5:6], v[1:4], s[8:15] dmask:0x5 ; encoding: [0x00,0x05,0x14,0xf0,0x01,0x05,0x02,0x00]

image_load_pck v5, v[1:4], s[8:15] dmask:0x1 glc
// GCN: image_load_pck v5, v[1:4], s[8:15] dmask:0x1 glc ; encoding: [0x00,0x21,0x08,0xf0,0x01,0x05,0x02,0x00]

image_load_pck_sgn v5, v[1:4], s[8:15] dmask:0x1 lwe
// GCN: image_load_pck_sgn v5, v[1:4], s[8:15] dmask:0x1 lwe ; encoding: [0x00,0x01,0x0e,0xf0,0x01,0x05,0x02,0x00]

image_load_mip_pck v5, v[1:4], s[8:15] dmask:0x1 d16
// NOSICI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// NOVI:   error: invalid operand for instruction
// NOGFX9: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

image_load_mip_pck v5, v[1:2], s[8:15] dmask:0x1 a16
// GFX9:   image_load_mip_pck v5, v[1:2], s[8:15] dmask:0x1 a16 ; encoding: [0x00,0x81,0x10,0xf0,0x01,0x05,0x02,0x00]
// NOSICI: :[[@LINE-2]]:{{[0-9]+}}: error: a16 modifier is not supported on this GPU
// NOVI:   error: a16 modifier is not supported on this GPU

image_store_mip_pck v252, v2, s[12:19] dmask:0x1 unorm
// GCN: image_store_mip_pck v252, v2, s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x2c,0xf0,0x02,0xfc,0x03,0x00]

image_store_mip_pck v252, v[2:3], s[12:19] dmask:0x1 unorm
// GCN: image_store_mip_pck v252, v[2:3], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x2c,0xf0,0x02,0xfc,0x03,0x00]

image_store_mip_pck v252, v[2:4], s[12:19] dmask:0x1 unorm
// GCN: image_store_mip_pck v252, v[2:4], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x2c,0xf0,0x02,0xfc,0x03,0x00]

image_store_mip_pck v252, v[2:5], s[12:19] dmask:0x1 unorm
// GCN: image_store_mip_pck v252, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x2c,0xf0,0x02,0xfc,0x03,0x00]

image_store_mip_pck v1, v[2:5], s[12:19] dmask:0x1 unorm glc slc lwe da
// GCN: image_store_mip_pck v1, v[2:5], s[12:19] dmask:0x1 unorm glc slc lwe da ; encoding: [0x00,0x71,0x2e,0xf2,0x02,0x01,0x03,0x00]

image_store_pck v1, v[2:5], s[12:19] dmask:0x1 unorm da
// GCN: image_store_pck v1, v[2:5], s[12:19] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x28,0xf0,0x02,0x01,0x03,0x00]

image_store_mip_pck v252, v[2:5], s[12:19] dmask:0x1 d16
// NOSICI: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// NOVI:   error: invalid operand for instruction
// NOGFX9: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand for instruction

image_store_mip_pck v252, v[2:3], s[12:19] dmask:0x1 a16
// GFX9:   image_store_mip_pck v252, v[2:3], s[12:19] dmask:0x1 a16 ; encoding: [0x00,0x81,0x2c,0xf0,0x02,0xfc,0x03,0x00]
// NOSICI: :[[@LINE-2]]:{{[0-9]+}}: error: a16 modifier is not supported on this GPU
// NOVI:   error: a16 modifier is not supported on this GPU

//===----------------------------------------------------------------------===//
// Image Sample
//===----------------------------------------------------------------------===//

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x7 unorm
// GCN: image_sample v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x7 unorm ; encoding: [0x00,0x17,0x80,0xf0,0xed,0xc1,0x27,0x00]

image_sample  v193, v237, s[28:35], s[4:7]
// GCN: image_sample v193, v237, s[28:35], s[4:7] ; encoding: [0x00,0x00,0x80,0xf0,0xed,0xc1,0x27,0x00]

image_sample  v193, v[237:238], s[28:35], s[4:7]
// GCN: image_sample v193, v[237:238], s[28:35], s[4:7] ; encoding: [0x00,0x00,0x80,0xf0,0xed,0xc1,0x27,0x00]

image_sample  v193, v[237:239], s[28:35], s[4:7]
// GCN: image_sample v193, v[237:239], s[28:35], s[4:7] ; encoding: [0x00,0x00,0x80,0xf0,0xed,0xc1,0x27,0x00]

image_sample  v193, v[237:240], s[28:35], s[4:7]
// GCN: image_sample v193, v[237:240], s[28:35], s[4:7] ; encoding: [0x00,0x00,0x80,0xf0,0xed,0xc1,0x27,0x00]

image_sample  v[193:194], v[237:240], s[28:35], s[4:7] tfe
// GCN: image_sample v[193:194], v[237:240], s[28:35], s[4:7] tfe ; encoding: [0x00,0x00,0x81,0xf0,0xed,0xc1,0x27,0x00]

// FIXME: This test is incorrect because r128 assumes a 128-bit SRSRC.
image_sample  v193, v[237:240], s[28:35], s[4:7] r128
// SICIVI: image_sample v193, v[237:240], s[28:35], s[4:7] r128 ; encoding: [0x00,0x80,0x80,0xf0,0xed,0xc1,0x27,0x00]
// NOGFX9: :[[@LINE-2]]:{{[0-9]+}}: error: r128 modifier is not supported on this GPU

image_sample  v193, v[237:240], s[28:35], s[4:7] d16
// NOSICI: :[[@LINE-1]]:{{[0-9]+}}: error: d16 modifier is not supported on this GPU
// GFX89:  image_sample v193, v[237:240], s[28:35], s[4:7] d16 ; encoding: [0x00,0x00,0x80,0xf0,0xed,0xc1,0x27,0x80]

//===----------------------------------------------------------------------===//
// Image Sample: d16 packed
//===----------------------------------------------------------------------===//

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x7 d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// GFX8_0:   image_sample v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x7 d16 ; encoding: [0x00,0x07,0x80,0xf0,0xed,0xc1,0x27,0x80]
// NOGFX8_1: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask, d16 and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe

//===----------------------------------------------------------------------===//
// Image Sample: d16 unpacked
//===----------------------------------------------------------------------===//

image_sample  v[193:194], v[237:240], s[28:35], s[4:7] dmask:0x7 d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// NOGFX8_0: :[[@LINE-2]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// GFX8_1:   image_sample v[193:194], v[237:240], s[28:35], s[4:7] dmask:0x7 d16 ; encoding: [0x00,0x07,0x80,0xf0,0xed,0xc1,0x27,0x80]
// GFX9:     image_sample v[193:194], v[237:240], s[28:35], s[4:7] dmask:0x7 d16 ; encoding: [0x00,0x07,0x80,0xf0,0xed,0xc1,0x27,0x80]

//===----------------------------------------------------------------------===//
// Image Sample: a16
//===----------------------------------------------------------------------===//

image_sample v[193:196], v[237:240], s[28:35], s[4:7] dmask:0xf a16
// GFX9:   image_sample v[193:196], v[237:240], s[28:35], s[4:7] dmask:0xf a16 ; encoding: [0x00,0x8f,0x80,0xf0,0xed,0xc1,0x27,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:   error: a16 modifier is not supported on this GPU

image_sample_d v[193:196], v[237:240], s[28:35], s[4:7] dmask:0xf a16
// GFX9:   image_sample_d v[193:196], v[237:240], s[28:35], s[4:7] dmask:0xf a16 ; encoding: [0x00,0x8f,0x88,0xf0,0xed,0xc1,0x27,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:   error: a16 modifier is not supported on this GPU

image_sample_c_d v[193:196], v[237:240], s[28:35], s[4:7] dmask:0xf a16
// GFX9:   image_sample_c_d v[193:196], v[237:240], s[28:35], s[4:7] dmask:0xf a16 ; encoding: [0x00,0x8f,0xa8,0xf0,0xed,0xc1,0x27,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:   error: a16 modifier is not supported on this GPU

image_sample_c_d_cl v[193:196], v[237:240], s[28:35], s[4:7] dmask:0xf a16
// GFX9:   image_sample_c_d_cl v[193:196], v[237:240], s[28:35], s[4:7] dmask:0xf a16 ; encoding: [0x00,0x8f,0xac,0xf0,0xed,0xc1,0x27,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:   error: a16 modifier is not supported on this GPU

//===----------------------------------------------------------------------===//
// Image Atomics
//===----------------------------------------------------------------------===//

image_atomic_add v4, v192, s[28:35] dmask:0x1 unorm glc
// SICI:  image_atomic_add v4, v192, s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x44,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_add v4, v192, s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x48,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_add v4, v[192:193], s[28:35] dmask:0x1 unorm glc
// SICI:  image_atomic_add v4, v[192:193], s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x44,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_add v4, v[192:193], s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x48,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_add v4, v[192:194], s[28:35] dmask:0x1 unorm glc
// SICI:  image_atomic_add v4, v[192:194], s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x44,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_add v4, v[192:194], s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x48,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_add v4, v[192:195], s[28:35] dmask:0x1 unorm glc
// SICI:  image_atomic_add v4, v[192:195], s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x44,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_add v4, v[192:195], s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x48,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_add v252, v2, s[8:15] dmask:0x1 unorm
// SICI:  image_atomic_add v252, v2, s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x44,0xf0,0x02,0xfc,0x02,0x00]
// GFX89: image_atomic_add v252, v2, s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x48,0xf0,0x02,0xfc,0x02,0x00]

image_atomic_add v[6:7], v255, s[8:15] dmask:0x3
// SICI:  image_atomic_add v[6:7], v255, s[8:15] dmask:0x3 ; encoding: [0x00,0x03,0x44,0xf0,0xff,0x06,0x02,0x00]
// GFX89: image_atomic_add v[6:7], v255, s[8:15] dmask:0x3 ; encoding: [0x00,0x03,0x48,0xf0,0xff,0x06,0x02,0x00]

image_atomic_add v7, v3, s[0:7] dmask:0x1 glc
// SICI:  image_atomic_add v7, v3, s[0:7] dmask:0x1 glc ; encoding: [0x00,0x21,0x44,0xf0,0x03,0x07,0x00,0x00]
// GFX89: image_atomic_add v7, v3, s[0:7] dmask:0x1 glc ; encoding: [0x00,0x21,0x48,0xf0,0x03,0x07,0x00,0x00]

image_atomic_add v8, v4, s[8:15] dmask:0x1 slc
// SICI:  image_atomic_add v8, v4, s[8:15] dmask:0x1 slc ; encoding: [0x00,0x01,0x44,0xf2,0x04,0x08,0x02,0x00]
// GFX89: image_atomic_add v8, v4, s[8:15] dmask:0x1 slc ; encoding: [0x00,0x01,0x48,0xf2,0x04,0x08,0x02,0x00]

image_atomic_add v9, v5, s[8:15] dmask:0x1 unorm glc slc lwe da
// SICI:  image_atomic_add v9, v5, s[8:15] dmask:0x1 unorm glc slc lwe da ; encoding: [0x00,0x71,0x46,0xf2,0x05,0x09,0x02,0x00]
// GFX89: image_atomic_add v9, v5, s[8:15] dmask:0x1 unorm glc slc lwe da ; encoding: [0x00,0x71,0x4a,0xf2,0x05,0x09,0x02,0x00]

image_atomic_add v10, v6, s[8:15] dmask:0x1 lwe
// SICI:  image_atomic_add v10, v6, s[8:15] dmask:0x1 lwe ; encoding: [0x00,0x01,0x46,0xf0,0x06,0x0a,0x02,0x00]
// GFX89: image_atomic_add v10, v6, s[8:15] dmask:0x1 lwe ; encoding: [0x00,0x01,0x4a,0xf0,0x06,0x0a,0x02,0x00]

image_atomic_add v11, v7, s[8:15] dmask:0x1 da
// SICI:  image_atomic_add v11, v7, s[8:15] dmask:0x1 da ; encoding: [0x00,0x41,0x44,0xf0,0x07,0x0b,0x02,0x00]
// GFX89: image_atomic_add v11, v7, s[8:15] dmask:0x1 da ; encoding: [0x00,0x41,0x48,0xf0,0x07,0x0b,0x02,0x00]

image_atomic_swap v4, v[192:195], s[28:35] dmask:0x1 unorm glc
// SICI:  image_atomic_swap v4, v[192:195], s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x3c,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_swap v4, v[192:195], s[28:35] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x40,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_cmpswap v[4:5], v[192:195], s[28:35] dmask:0x3 unorm glc
// SICI:  image_atomic_cmpswap v[4:5], v[192:195], s[28:35] dmask:0x3 unorm glc ; encoding: [0x00,0x33,0x40,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_cmpswap v[4:5], v[192:195], s[28:35] dmask:0x3 unorm glc ; encoding: [0x00,0x33,0x44,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xf unorm glc
// SICI:  image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xf unorm glc ; encoding: [0x00,0x3f,0x40,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xf unorm glc ; encoding: [0x00,0x3f,0x44,0xf0,0xc0,0x04,0x07,0x00]

; Test dmask + tfe for image_atomic instructions
image_atomic_add v4, v10, s[8:15] dmask:0x1
// SICI:  image_atomic_add v4, v10, s[8:15] dmask:0x1 ; encoding: [0x00,0x01,0x44,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_add v4, v10, s[8:15] dmask:0x1 ; encoding: [0x00,0x01,0x48,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_add v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_add v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x45,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_add v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x49,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_add v[4:5], v10, s[8:15] dmask:0x3
// SICI:  image_atomic_add v[4:5], v10, s[8:15] dmask:0x3 ; encoding: [0x00,0x03,0x44,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_add v[4:5], v10, s[8:15] dmask:0x3 ; encoding: [0x00,0x03,0x48,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_add v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_add v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x45,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_add v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x49,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_swap v4, v[192:195], s[28:35] dmask:0x1
// SICI:  image_atomic_swap v4, v[192:195], s[28:35] dmask:0x1 ; encoding: [0x00,0x01,0x3c,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_swap v4, v[192:195], s[28:35] dmask:0x1 ; encoding: [0x00,0x01,0x40,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_swap v[4:5], v[192:195], s[28:35] dmask:0x1 tfe
// SICI:  image_atomic_swap v[4:5], v[192:195], s[28:35] dmask:0x1 tfe ; encoding: [0x00,0x01,0x3d,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_swap v[4:5], v[192:195], s[28:35] dmask:0x1 tfe ; encoding: [0x00,0x01,0x41,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_swap v[4:5], v[192:195], s[28:35] dmask:0x3
// SICI:  image_atomic_swap v[4:5], v[192:195], s[28:35] dmask:0x3 ; encoding: [0x00,0x03,0x3c,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_swap v[4:5], v[192:195], s[28:35] dmask:0x3 ; encoding: [0x00,0x03,0x40,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_swap v[4:6], v[192:195], s[28:35] dmask:0x3 tfe
// SICI:  image_atomic_swap v[4:6], v[192:195], s[28:35] dmask:0x3 tfe ; encoding: [0x00,0x03,0x3d,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_swap v[4:6], v[192:195], s[28:35] dmask:0x3 tfe ; encoding: [0x00,0x03,0x41,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_sub v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_sub v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x49,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_sub v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x4d,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_sub v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_sub v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x49,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_sub v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x4d,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_smin v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_smin v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x51,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_smin v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x51,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_smin v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_smin v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x51,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_smin v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x51,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_umin v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_umin v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x55,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_umin v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x55,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_umin v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_umin v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x55,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_umin v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x55,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_smax v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_smax v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x59,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_smax v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x59,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_smax v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_smax v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x59,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_smax v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x59,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_umax v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_umax v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x5d,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_umax v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x5d,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_umax v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_umax v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x5d,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_umax v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x5d,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_and v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_and v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x61,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_and v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x61,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_and v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_and v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x61,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_and v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x61,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_or v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_or v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x65,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_or v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x65,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_or v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_or v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x65,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_or v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x65,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_xor v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_xor v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x69,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_xor v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x69,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_xor v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_xor v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x69,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_xor v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x69,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_inc v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_inc v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x6d,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_inc v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x6d,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_inc v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_inc v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x6d,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_inc v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x6d,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_dec v[4:5], v10, s[8:15] dmask:0x1 tfe
// SICI:  image_atomic_dec v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x71,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_dec v[4:5], v10, s[8:15] dmask:0x1 tfe ; encoding: [0x00,0x01,0x71,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_dec v[4:6], v10, s[8:15] dmask:0x3 tfe
// SICI:  image_atomic_dec v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x71,0xf0,0x0a,0x04,0x02,0x00]
// GFX89: image_atomic_dec v[4:6], v10, s[8:15] dmask:0x3 tfe ; encoding: [0x00,0x03,0x71,0xf0,0x0a,0x04,0x02,0x00]

image_atomic_cmpswap v[4:5], v[192:195], s[28:35] dmask:0x3
// SICI:  image_atomic_cmpswap v[4:5], v[192:195], s[28:35] dmask:0x3 ; encoding: [0x00,0x03,0x40,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_cmpswap v[4:5], v[192:195], s[28:35] dmask:0x3 ; encoding: [0x00,0x03,0x44,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_cmpswap v[4:6], v[192:195], s[28:35] dmask:0x3 tfe
// SICI:  image_atomic_cmpswap v[4:6], v[192:195], s[28:35] dmask:0x3 tfe ; encoding: [0x00,0x03,0x41,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_cmpswap v[4:6], v[192:195], s[28:35] dmask:0x3 tfe ; encoding: [0x00,0x03,0x45,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xf
// SICI:  image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xf ; encoding: [0x00,0x0f,0x40,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_cmpswap v[4:7], v[192:195], s[28:35] dmask:0xf ; encoding: [0x00,0x0f,0x44,0xf0,0xc0,0x04,0x07,0x00]

image_atomic_cmpswap v[4:8], v[192:195], s[28:35] dmask:0xf tfe
// SICI:  image_atomic_cmpswap v[4:8], v[192:195], s[28:35] dmask:0xf tfe ; encoding: [0x00,0x0f,0x41,0xf0,0xc0,0x04,0x07,0x00]
// GFX89: image_atomic_cmpswap v[4:8], v[192:195], s[28:35] dmask:0xf tfe ; encoding: [0x00,0x0f,0x45,0xf0,0xc0,0x04,0x07,0x00]

// FIXME: This test is incorrect because r128 assumes a 128-bit SRSRC.
image_atomic_add v10, v6, s[8:15] dmask:0x1 r128
// SICI: image_atomic_add v10, v6, s[8:15] dmask:0x1 r128 ; encoding: [0x00,0x81,0x44,0xf0,0x06,0x0a,0x02,0x00]
// VI:   image_atomic_add v10, v6, s[8:15] dmask:0x1 r128 ; encoding: [0x00,0x81,0x48,0xf0,0x06,0x0a,0x02,0x00]
// NOGFX9: :[[@LINE-3]]:{{[0-9]+}}: error: r128 modifier is not supported on this GPU

//===----------------------------------------------------------------------===//
// Image Gather4
//===----------------------------------------------------------------------===//

image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1
// GCN: image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 ; encoding: [0x00,0x01,0x00,0xf1,0x01,0x05,0x62,0x00]

image_gather4 v[5:8], v[1:2], s[8:15], s[12:15] dmask:0x1
// GCN: image_gather4 v[5:8], v[1:2], s[8:15], s[12:15] dmask:0x1 ; encoding: [0x00,0x01,0x00,0xf1,0x01,0x05,0x62,0x00]

image_gather4 v[5:8], v[1:3], s[8:15], s[12:15] dmask:0x1
// GCN: image_gather4 v[5:8], v[1:3], s[8:15], s[12:15] dmask:0x1 ; encoding: [0x00,0x01,0x00,0xf1,0x01,0x05,0x62,0x00]

image_gather4 v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x2
// GCN: image_gather4 v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x2 ; encoding: [0x00,0x02,0x00,0xf1,0x01,0x05,0x62,0x00]

image_gather4 v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x4
// GCN: image_gather4 v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x4 ; encoding: [0x00,0x04,0x00,0xf1,0x01,0x05,0x62,0x00]

image_gather4 v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x8
// GCN: image_gather4 v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x8 ; encoding: [0x00,0x08,0x00,0xf1,0x01,0x05,0x62,0x00]

image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// GFX8_0:   image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 d16 ; encoding: [0x00,0x01,0x00,0xf1,0x01,0x05,0x62,0x80]
// NOGFX8_1: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask, d16 and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe

image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16
// NOSICI:   error: d16 modifier is not supported on this GPU
// NOGFX8_0: :[[@LINE-2]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// GFX8_1:   image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16 ; encoding: [0x00,0x01,0x00,0xf1,0x01,0x05,0x62,0x80]
// GFX9:     image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16 ; encoding: [0x00,0x01,0x00,0xf1,0x01,0x05,0x62,0x80]

image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1
// NOSICI:   error: image data size does not match dmask and tfe
// NOGFX8_0: :[[@LINE-2]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// NOGFX8_1: :[[@LINE-3]]:{{[0-9]+}}: error: image data size does not match dmask, d16 and tfe
// NOGFX9:   error: image data size does not match dmask, d16 and tfe

image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 a16
// GFX9:     image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 a16 ; encoding: [0x00,0x81,0x00,0xf1,0x01,0x05,0x62,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:   error: a16 modifier is not supported on this GPU

image_gather4_b_cl v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 a16
// GFX9:    image_gather4_b_cl v[5:8], v[1:4], s[8:15], s[12:15] dmask:0x1 a16 ; encoding: [0x00,0x81,0x18,0xf1,0x01,0x05,0x62,0x00]
// NOSICI:   error: a16 modifier is not supported on this GPU
// NOVI:   error: a16 modifier is not supported on this GPU
