// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s --check-prefix=GCN
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx900 -filetype=obj %s | llvm-objcopy -S -K keep_symbol - | llvm-objdump -disassemble -mcpu=gfx900 - | FileCheck %s --check-prefix=BIN

// GCN: s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]
// BIN: s_endpgm    // 000000000000: BF810000
s_endpgm

// GCN: s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]
// BIN: s_endpgm    // 000000000004: BF810000
s_endpgm 0

  
// GCN: s_endpgm 1 ; encoding: [0x01,0x00,0x81,0xbf]
// BIN: s_endpgm 1  // 000000000008: BF810001
s_endpgm 1
  
  
