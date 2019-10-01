// RUN: llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck %s --check-prefix=CI

v_readfirstlane_b32 s0, lds_direct
// CI: v_readfirstlane_b32 s0, src_lds_direct ; encoding: [0xfe,0x04,0x00,0x7e]

v_readlane_b32 s0, lds_direct, s0
// CI: v_readlane_b32 s0, src_lds_direct, s0 ; encoding: [0xfe,0x00,0x00,0x02]

v_writelane_b32 v0, lds_direct, s0
// CI: v_writelane_b32 v0, src_lds_direct, s0 ; encoding: [0xfe,0x00,0x00,0x04]
