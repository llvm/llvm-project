// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=SI  %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=SI %s
// RUN: llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=CI %s

// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck %s --check-prefix=NOSI
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI  %s 2>&1 | FileCheck %s --check-prefix=NOSI
//===----------------------------------------------------------------------===//
// Offset Handling
//===----------------------------------------------------------------------===//

s_load_dword s1, s[2:3], 0xfc
// GCN: s_load_dword s1, s[2:3], 0xfc ; encoding: [0xfc,0x83,0x00,0xc0]

s_load_dword s1, s[2:3], 0xff
// GCN: s_load_dword s1, s[2:3], 0xff ; encoding: [0xff,0x83,0x00,0xc0]

s_load_dword s1, s[2:3], 0x100
// NOSI: error: instruction not supported on this GPU
// CI: s_load_dword s1, s[2:3], 0x100 ; encoding: [0xff,0x82,0x00,0xc0,0x00,0x01,0x00,0x00]

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

s_load_dword s1, s[2:3], 1
// GCN: s_load_dword s1, s[2:3], 0x1 ; encoding: [0x01,0x83,0x00,0xc0]

s_load_dword s1, s[2:3], s4
// GCN: s_load_dword s1, s[2:3], s4 ; encoding: [0x04,0x82,0x00,0xc0]

s_load_dwordx2 s[2:3], s[2:3], 1
// GCN: s_load_dwordx2 s[2:3], s[2:3], 0x1 ; encoding: [0x01,0x03,0x41,0xc0]

s_load_dwordx2 s[2:3], s[2:3], s4
// GCN: s_load_dwordx2 s[2:3], s[2:3], s4 ; encoding: [0x04,0x02,0x41,0xc0]

s_load_dwordx4 s[4:7], s[2:3], 1
// GCN: s_load_dwordx4 s[4:7], s[2:3], 0x1 ; encoding: [0x01,0x03,0x82,0xc0]

s_load_dwordx4 s[4:7], s[2:3], s4
// GCN: s_load_dwordx4 s[4:7], s[2:3], s4 ; encoding: [0x04,0x02,0x82,0xc0]

s_load_dwordx4 s[100:103], s[2:3], s4
// GCN: s_load_dwordx4 s[100:103], s[2:3], s4 ; encoding: [0x04,0x02,0xb2,0xc0]

s_load_dwordx8 s[8:15], s[2:3], 1
// GCN: s_load_dwordx8 s[8:15], s[2:3], 0x1 ; encoding: [0x01,0x03,0xc4,0xc0]

s_load_dwordx8 s[8:15], s[2:3], s4
// GCN: s_load_dwordx8 s[8:15], s[2:3], s4 ; encoding: [0x04,0x02,0xc4,0xc0]

s_load_dwordx8 s[96:103], s[2:3], s4
// GCN: s_load_dwordx8 s[96:103], s[2:3], s4 ; encoding: [0x04,0x02,0xf0,0xc0]

s_load_dwordx16 s[16:31], s[2:3], 1
// GCN: s_load_dwordx16 s[16:31], s[2:3], 0x1 ; encoding: [0x01,0x03,0x08,0xc1]

s_load_dwordx16 s[16:31], s[2:3], s4
// GCN: s_load_dwordx16 s[16:31], s[2:3], s4 ; encoding: [0x04,0x02,0x08,0xc1]

s_load_dwordx16 s[88:103], s[2:3], s4
// GCN: s_load_dwordx16 s[88:103], s[2:3], s4 ; encoding: [0x04,0x02,0x2c,0xc1]

s_dcache_inv
// GCN: s_dcache_inv ; encoding: [0x00,0x00,0xc0,0xc7]

s_dcache_inv_vol
// CI: s_dcache_inv_vol ; encoding: [0x00,0x00,0x40,0xc7]
// NOSI: error: instruction not supported on this GPU

s_memtime s[0:1]
// GCN: s_memtime s[0:1] ; encoding: [0x00,0x00,0x80,0xc7]
