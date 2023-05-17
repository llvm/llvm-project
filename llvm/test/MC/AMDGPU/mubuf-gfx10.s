// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10 %s

buffer_load_sbyte off, s[8:11], s3 glc slc lds
// GFX10: buffer_load_sbyte off, s[8:11], s3 glc slc lds ; encoding: [0x00,0x40,0x25,0xe0,0x00,0x00,0x42,0x03]

buffer_load_sbyte off, s[8:11], s3 glc slc dlc lds
// GFX10: buffer_load_sbyte off, s[8:11], s3 glc slc dlc lds ; encoding: [0x00,0xc0,0x25,0xe0,0x00,0x00,0x42,0x03]

buffer_load_sbyte v5, off, s[8:11], s3 glc slc dlc
// GFX10: buffer_load_sbyte v5, off, s[8:11], s3 glc slc dlc ; encoding: [0x00,0xc0,0x24,0xe0,0x00,0x05,0x42,0x03]

buffer_load_sbyte v[5:6], off, s[8:11], s3 glc slc dlc tfe
// GFX10: buffer_load_sbyte v[5:6], off, s[8:11], s3 glc slc dlc tfe ; encoding: [0x00,0xc0,0x24,0xe0,0x00,0x05,0xc2,0x03]

buffer_atomic_fcmpswap v[0:1], off, s[0:3], s0 offset:4095
// GFX10: buffer_atomic_fcmpswap v[0:1], off, s[0:3], s0 offset:4095 ; encoding: [0xff,0x0f,0xf8,0xe0,0x00,0x00,0x00,0x00]

buffer_atomic_fcmpswap_x2 v[0:3], off, s[0:3], s0 offset:4095
// GFX10: buffer_atomic_fcmpswap_x2 v[0:3], off, s[0:3], s0 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe1,0x00,0x00,0x00,0x00]

buffer_atomic_fcmpswap_x2 v[0:3], v0, s[0:3], s0 idxen offset:4095
// GFX10: buffer_atomic_fcmpswap_x2 v[0:3], v0, s[0:3], s0 idxen offset:4095 ; encoding: [0xff,0x2f,0x78,0xe1,0x00,0x00,0x00,0x00]

buffer_atomic_fmax v1, off, s[0:3], s0 offset:4095
// GFX10: buffer_atomic_fmax v1, off, s[0:3], s0 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0x01,0x00,0x00]

buffer_atomic_fmax v0, off, s[0:3], s0 offset:7
// GFX10: buffer_atomic_fmax v0, off, s[0:3], s0 offset:7 ; encoding: [0x07,0x00,0x00,0xe1,0x00,0x00,0x00,0x00]

buffer_atomic_fmax v0, off, s[0:3], s0 offset:4095 glc
// GFX10: buffer_atomic_fmax v0, off, s[0:3], s0 offset:4095 glc ; encoding: [0xff,0x4f,0x00,0xe1,0x00,0x00,0x00,0x00]

buffer_atomic_fmax_x2 v[5:6], off, s[0:3], s0 offset:4095
// GFX10: buffer_atomic_fmax_x2 v[5:6], off, s[0:3], s0 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe1,0x00,0x05,0x00,0x00]

buffer_atomic_fmax_x2 v[0:1], v0, s[0:3], s0 idxen offset:4095
// GFX10: buffer_atomic_fmax_x2 v[0:1], v0, s[0:3], s0 idxen offset:4095 ; encoding: [0xff,0x2f,0x80,0xe1,0x00,0x00,0x00,0x00]

buffer_atomic_fmin v0, off, s[0:3], s0
// GFX10: buffer_atomic_fmin v0, off, s[0:3], s0 ; encoding: [0x00,0x00,0xfc,0xe0,0x00,0x00,0x00,0x00]

buffer_atomic_fmin v0, off, s[0:3], s0 offset:0
// GFX10: buffer_atomic_fmin v0, off, s[0:3], s0 ; encoding: [0x00,0x00,0xfc,0xe0,0x00,0x00,0x00,0x00]

buffer_atomic_fmin_x2 v[0:1], off, s[0:3], s0 offset:4095 slc
// GFX10: buffer_atomic_fmin_x2 v[0:1], off, s[0:3], s0 offset:4095 slc ; encoding: [0xff,0x0f,0x7c,0xe1,0x00,0x00,0x40,0x00]

buffer_atomic_fmin_x2 v[0:1], v0, s[0:3], s0 idxen offset:4095
// GFX10: buffer_atomic_fmin_x2 v[0:1], v0, s[0:3], s0 idxen offset:4095 ; encoding: [0xff,0x2f,0x7c,0xe1,0x00,0x00,0x00,0x00]
