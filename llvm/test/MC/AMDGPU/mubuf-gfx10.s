// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10 %s

buffer_load_sbyte v5, off, s[8:11], s3 glc slc lds
// GFX10: buffer_load_sbyte v5, off, s[8:11], s3 glc slc lds ; encoding: [0x00,0x40,0x25,0xe0,0x00,0x05,0x42,0x03]

buffer_load_sbyte v5, off, s[8:11], s3 glc slc lds dlc
// GFX10: buffer_load_sbyte v5, off, s[8:11], s3 glc slc lds dlc ; encoding: [0x00,0xc0,0x25,0xe0,0x00,0x05,0x42,0x03]

buffer_load_sbyte v5, off, s[8:11], s3 glc slc dlc
// GFX10: buffer_load_sbyte v5, off, s[8:11], s3 glc slc dlc ; encoding: [0x00,0xc0,0x24,0xe0,0x00,0x05,0x42,0x03]
