// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=NOSI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck -check-prefix=NOSI %s

s_dcache_wb
// VI: s_dcache_wb  ; encoding: [0x00,0x00,0x84,0xc0,0x00,0x00,0x00,0x00]
// NOSI: error: instruction not supported on this GPU

s_dcache_wb_vol
// VI: s_dcache_wb_vol ; encoding: [0x00,0x00,0x8c,0xc0,0x00,0x00,0x00,0x00]
// NOSI: error: instruction not supported on this GPU

s_memrealtime s[4:5]
// VI: s_memrealtime s[4:5] ; encoding: [0x00,0x01,0x94,0xc0,0x00,0x00,0x00,0x00]
// NOSI: error: instruction not supported on this GPU

// FIXME: Should error about instruction on GPU
s_store_dword s1, s[2:3], 0xfc
// VI: s_store_dword s1, s[2:3], 0xfc  ; encoding: [0x41,0x00,0x42,0xc0,0xfc,0x00,0x00,0x00]
// NOSI: error: instruction not supported on this GPU

s_store_dword s1, s[2:3], 0xfc glc
// VI: s_store_dword s1, s[2:3], 0xfc glc ; encoding: [0x41,0x00,0x43,0xc0,0xfc,0x00,0x00,0x00]
// NOSI: error: invalid operand for instruction

s_store_dword s1, s[2:3], s4
// VI: s_store_dword s1, s[2:3], s4    ; encoding: [0x41,0x00,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSI: error: instruction not supported on this GPU

s_store_dword s1, s[2:3], s4 glc
// VI: s_store_dword s1, s[2:3], s4 glc ; encoding: [0x41,0x00,0x41,0xc0,0x04,0x00,0x00,0x00]
// NOSI: error: invalid operand for instruction

// FIXME: Should error on SI instead of silently ignoring glc
s_load_dword s1, s[2:3], 0xfc glc
// VI: s_load_dword s1, s[2:3], 0xfc glc ; encoding: [0x41,0x00,0x03,0xc0,0xfc,0x00,0x00,0x00]

s_load_dword s1, s[2:3], s4 glc
// VI: s_load_dword s1, s[2:3], s4 glc ; encoding: [0x41,0x00,0x01,0xc0,0x04,0x00,0x00,0x00]
