// RUN: llvm-mc -triple=amdgcn -show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple=amdgcn -mcpu=generic -show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=generic-hsa -show-encoding %s | FileCheck %s

// Targets without a real CPU use the SI s_waitcnt layout.

s_waitcnt vmcnt(0) lgkmcnt(0)
// CHECK: s_waitcnt vmcnt(0) lgkmcnt(0) ; encoding: [0x70,0x00,0x8c,0xbf]

s_waitcnt expcnt(0)
// CHECK: s_waitcnt expcnt(0) ; encoding: [0x0f,0x0f,0x8c,0xbf]
