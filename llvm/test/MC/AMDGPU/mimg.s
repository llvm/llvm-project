// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s --check-prefix=SICI
// RUN: llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck %s --check-prefix=SICI
// RUN: llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck %s --check-prefix=VI

image_load    v[4:6], v[237:240], s[28:35] dmask:0x7 unorm
// SICI: image_load v[4:6], v[237:240], s[28:35] dmask:0x7 unorm ; encoding: [0x00,0x17,0x00,0xf0,0xed,0x04,0x07,0x00]
// VI:   image_load v[4:6], v[237:240], s[28:35] dmask:0x7 unorm ; encoding: [0x00,0x17,0x00,0xf0,0xed,0x04,0x07,0x00]

image_store   v[193:195], v[237:240], s[28:35] dmask:0x7 unorm
// SICI: image_store v[193:195], v[237:240], s[28:35] dmask:0x7 unorm ; encoding: [0x00,0x17,0x20,0xf0,0xed,0xc1,0x07,0x00]
// VI  : image_store v[193:195], v[237:240], s[28:35] dmask:0x7 unorm ; encoding: [0x00,0x17,0x20,0xf0,0xed,0xc1,0x07,0x00]

image_sample  v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x7 unorm
// SICI: image_sample v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x7 unorm ; encoding: [0x00,0x17,0x80,0xf0,0xed,0xc1,0x27,0x00]
// VI  : image_sample v[193:195], v[237:240], s[28:35], s[4:7] dmask:0x7 unorm ; encoding: [0x00,0x17,0x80,0xf0,0xed,0xc1,0x27,0x00]
