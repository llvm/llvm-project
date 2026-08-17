// RUN: llvm-mc -triple=amdgpu6.00 %s | FileCheck -strict-whitespace %s -check-prefix=WHITESPACE
// RUN: llvm-mc -triple=amdgpu6.00 -show-encoding %s | FileCheck %s --check-prefix=GCN
// RUN: llvm-mc -triple=amdgpu9.00 -filetype=obj %s | llvm-objcopy -S -K keep_symbol - | llvm-objdump -d - | FileCheck %s --check-prefix=BIN

// WHITESPACE: s_endpgm{{$}}
// GCN: s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]
// BIN: s_endpgm    // 000000000000: BF810000
s_endpgm

// WHITESPACE: s_endpgm{{$}}
// GCN: s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]
// BIN: s_endpgm    // 000000000004: BF810000
s_endpgm 0

// WHITESPACE: s_endpgm 1{{$}}
// GCN: s_endpgm 1 ; encoding: [0x01,0x00,0x81,0xbf]
// BIN: s_endpgm 1  // 000000000008: BF810001
s_endpgm 1
