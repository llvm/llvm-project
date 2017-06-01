// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=kaveri -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=SICI %s
// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=NOSICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck -check-prefix=NOSICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=NOSICI %s

s_dcache_wb
// VI: s_dcache_wb  ; encoding: [0x00,0x00,0x84,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_dcache_wb_vol
// VI: s_dcache_wb_vol ; encoding: [0x00,0x00,0x8c,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_memrealtime s[4:5]
// VI: s_memrealtime s[4:5] ; encoding: [0x00,0x01,0x94,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_memrealtime tba
// VI: s_memrealtime tba ; encoding: [0x00,0x1b,0x94,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_memrealtime tma
// VI: s_memrealtime tma ; encoding: [0x80,0x1b,0x94,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_memrealtime ttmp[0:1]
// VI: s_memrealtime ttmp[0:1] ; encoding: [0x00,0x1c,0x94,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

// FIXME: Should error about instruction on GPU
s_store_dword s1, s[2:3], 0xfc
// VI: s_store_dword s1, s[2:3], 0xfc  ; encoding: [0x41,0x00,0x42,0xc0,0xfc,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_store_dword s1, s[2:3], 0xfc glc
// VI: s_store_dword s1, s[2:3], 0xfc glc ; encoding: [0x41,0x00,0x43,0xc0,0xfc,0x00,0x00,0x00]
// NOSICI: error: invalid operand for instruction

s_store_dword s1, s[2:3], s4
// VI: s_store_dword s1, s[2:3], s4 ; encoding: [0x41,0x00,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_store_dword s1, s[2:3], s4 glc
// VI: s_store_dword s1, s[2:3], s4 glc ; encoding: [0x41,0x00,0x41,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: invalid operand for instruction

s_store_dword tba_lo, s[2:3], s4
// VI: s_store_dword tba_lo, s[2:3], s4 ; encoding: [0x01,0x1b,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_store_dword tba_hi, s[2:3], s4
// VI: s_store_dword tba_hi, s[2:3], s4 ; encoding: [0x41,0x1b,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_store_dword tma_lo, s[2:3], s4
// VI: s_store_dword tma_lo, s[2:3], s4 ; encoding: [0x81,0x1b,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_store_dword tma_hi, s[2:3], s4
// VI: s_store_dword tma_hi, s[2:3], s4 ; encoding: [0xc1,0x1b,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

// FIXME: Should error on SI instead of silently ignoring glc
s_load_dword s1, s[2:3], 0xfc glc
// VI: s_load_dword s1, s[2:3], 0xfc glc ; encoding: [0x41,0x00,0x03,0xc0,0xfc,0x00,0x00,0x00]

s_load_dword s1, s[2:3], s4 glc
// VI: s_load_dword s1, s[2:3], s4 glc ; encoding: [0x41,0x00,0x01,0xc0,0x04,0x00,0x00,0x00]

s_buffer_store_dword s10, s[92:95], m0
// VI: s_buffer_store_dword s10, s[92:95], m0 ; encoding: [0xae,0x02,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dword tba_lo, s[92:95], m0
// VI: s_buffer_store_dword tba_lo, s[92:95], m0 ; encoding: [0x2e,0x1b,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dword tba_hi, s[92:95], m0
// VI: s_buffer_store_dword tba_hi, s[92:95], m0 ; encoding: [0x6e,0x1b,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dword tma_lo, s[92:95], m0
// VI: s_buffer_store_dword tma_lo, s[92:95], m0 ; encoding: [0xae,0x1b,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dword tma_hi, s[92:95], m0
// VI: s_buffer_store_dword tma_hi, s[92:95], m0 ; encoding: [0xee,0x1b,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dword ttmp0, s[92:95], m0
// VI: s_buffer_store_dword ttmp0, s[92:95], m0 ; encoding: [0x2e,0x1c,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[10:11], s[92:95], m0
// VI: s_buffer_store_dwordx2 s[10:11], s[92:95], m0 ; encoding: [0xae,0x02,0x64,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[8:11], s[92:95], m0 glc
// VI: s_buffer_store_dwordx4 s[8:11], s[92:95], m0 glc ; encoding: [0x2e,0x02,0x69,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: invalid operand for instruction

s_buffer_store_dwordx2 tba, s[92:95], m0 glc
// VI: s_buffer_store_dwordx2 tba, s[92:95], m0 glc ; encoding: [0x2e,0x1b,0x65,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: invalid operand for instruction

s_buffer_load_dword s10, s[92:95], m0
// VI: s_buffer_load_dword s10, s[92:95], m0 ; encoding: [0xae,0x02,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword s10, s[92:95], m0 ; encoding: [0x7c,0x5c,0x05,0xc2]

s_buffer_load_dword tba_lo, s[92:95], m0
// VI: s_buffer_load_dword tba_lo, s[92:95], m0 ; encoding: [0x2e,0x1b,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword tba_lo, s[92:95], m0 ; encoding: [0x7c,0x5c,0x36,0xc2]

s_buffer_load_dword tba_hi, s[92:95], m0
// VI: s_buffer_load_dword tba_hi, s[92:95], m0 ; encoding: [0x6e,0x1b,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword tba_hi, s[92:95], m0 ; encoding: [0x7c,0xdc,0x36,0xc2]

s_buffer_load_dword tma_lo, s[92:95], m0
// VI: s_buffer_load_dword tma_lo, s[92:95], m0 ; encoding: [0xae,0x1b,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword tma_lo, s[92:95], m0 ; encoding: [0x7c,0x5c,0x37,0xc2]

s_buffer_load_dword tma_hi, s[92:95], m0
// VI: s_buffer_load_dword tma_hi, s[92:95], m0 ; encoding: [0xee,0x1b,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword tma_hi, s[92:95], m0 ; encoding: [0x7c,0xdc,0x37,0xc2]

s_buffer_load_dword ttmp0, s[92:95], m0
// VI: s_buffer_load_dword ttmp0, s[92:95], m0 ; encoding: [0x2e,0x1c,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword ttmp0, s[92:95], m0 ; encoding: [0x7c,0x5c,0x38,0xc2]

s_buffer_load_dwordx2 s[10:11], s[92:95], m0
// VI: s_buffer_load_dwordx2 s[10:11], s[92:95], m0 ; encoding: [0xae,0x02,0x24,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dwordx2 s[10:11], s[92:95], m0 ; encoding: [0x7c,0x5c,0x45,0xc2]

s_buffer_load_dwordx2 tba, s[92:95], m0
// VI: s_buffer_load_dwordx2 tba, s[92:95], m0 ; encoding: [0x2e,0x1b,0x24,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dwordx2 tba, s[92:95], m0 ; encoding: [0x7c,0x5c,0x76,0xc2]

s_buffer_load_dwordx2 tma, s[92:95], m0
// VI: s_buffer_load_dwordx2 tma, s[92:95], m0 ; encoding: [0xae,0x1b,0x24,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dwordx2 tma, s[92:95], m0 ; encoding: [0x7c,0x5c,0x77,0xc2]

s_buffer_load_dwordx2 ttmp[0:1], s[92:95], m0
// VI: s_buffer_load_dwordx2 ttmp[0:1], s[92:95], m0 ; encoding: [0x2e,0x1c,0x24,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dwordx2 ttmp[0:1], s[92:95], m0 ; encoding: [0x7c,0x5c,0x78,0xc2]

// FIXME: Should error on SI instead of silently ignoring glc
s_buffer_load_dwordx4 s[8:11], s[92:95], m0 glc
// VI: s_buffer_load_dwordx4 s[8:11], s[92:95], m0 glc ; encoding: [0x2e,0x02,0x29,0xc0,0x7c,0x00,0x00,0x00]
