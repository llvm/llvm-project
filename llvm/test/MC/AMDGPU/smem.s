// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=kaveri -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=GFX89 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=GFX89 -check-prefix=GFX9 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=NOSICI -check-prefix=NOSICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck -check-prefix=NOSICI -check-prefix=NOSICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=kaveri %s 2>&1 | FileCheck -check-prefix=NOSICI -check-prefix=NOSICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=NOSICIVI -check-prefix=NOVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=NOGFX9 %s

s_dcache_wb
// GFX89: s_dcache_wb  ; encoding: [0x00,0x00,0x84,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_dcache_wb_vol
// GFX89: s_dcache_wb_vol ; encoding: [0x00,0x00,0x8c,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_atc_probe 0x7, s[4:5], s0
// GFX89:  s_atc_probe 7, s[4:5], s0 ; encoding: [0xc2,0x01,0x98,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_atc_probe 0x0, s[4:5], 0x0
// GFX89:  s_atc_probe 0, s[4:5], 0x0 ; encoding: [0x02,0x00,0x9a,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_atc_probe_buffer 0x1, s[8:11], s0
// GFX89:  s_atc_probe_buffer 1, s[8:11], s0 ; encoding: [0x44,0x00,0x9c,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_atc_probe_buffer 0x0, s[8:11], s101
// GFX89:  s_atc_probe_buffer 0, s[8:11], s101 ; encoding: [0x04,0x00,0x9c,0xc0,0x65,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_memrealtime s[4:5]
// GFX89: s_memrealtime s[4:5] ; encoding: [0x00,0x01,0x94,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_memrealtime tba
// VI: s_memrealtime tba ; encoding: [0x00,0x1b,0x94,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

s_memrealtime tma
// VI: s_memrealtime tma ; encoding: [0x80,0x1b,0x94,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

s_memrealtime ttmp[0:1]
// VI:   s_memrealtime ttmp[0:1] ; encoding: [0x00,0x1c,0x94,0xc0,0x00,0x00,0x00,0x00]
// GFX9: s_memrealtime ttmp[0:1] ; encoding: [0x00,0x1b,0x94,0xc0,0x00,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

// FIXME: Should error about instruction on GPU
s_store_dword s1, s[2:3], 0xfc
// GFX89: s_store_dword s1, s[2:3], 0xfc  ; encoding: [0x41,0x00,0x42,0xc0,0xfc,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_store_dword s1, s[2:3], 0xfc glc
// GFX89: s_store_dword s1, s[2:3], 0xfc glc ; encoding: [0x41,0x00,0x43,0xc0,0xfc,0x00,0x00,0x00]
// NOSICI: error: invalid operand for instruction

s_store_dword s1, s[2:3], s4
// GFX89: s_store_dword s1, s[2:3], s4 ; encoding: [0x41,0x00,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_store_dword s1, s[2:3], s4 glc
// GFX89: s_store_dword s1, s[2:3], s4 glc ; encoding: [0x41,0x00,0x41,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: invalid operand for instruction

s_store_dword tba_lo, s[2:3], s4
// VI: s_store_dword tba_lo, s[2:3], s4 ; encoding: [0x01,0x1b,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

s_store_dword tba_hi, s[2:3], s4
// VI: s_store_dword tba_hi, s[2:3], s4 ; encoding: [0x41,0x1b,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

s_store_dword tma_lo, s[2:3], s4
// VI: s_store_dword tma_lo, s[2:3], s4 ; encoding: [0x81,0x1b,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

s_store_dword tma_hi, s[2:3], s4
// VI: s_store_dword tma_hi, s[2:3], s4 ; encoding: [0xc1,0x1b,0x40,0xc0,0x04,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

// FIXME: Should error on SI instead of silently ignoring glc
s_load_dword s1, s[2:3], 0xfc glc
// GFX89: s_load_dword s1, s[2:3], 0xfc glc ; encoding: [0x41,0x00,0x03,0xc0,0xfc,0x00,0x00,0x00]

s_load_dword s1, s[2:3], s4 glc
// GFX89: s_load_dword s1, s[2:3], s4 glc ; encoding: [0x41,0x00,0x01,0xc0,0x04,0x00,0x00,0x00]

s_buffer_store_dword s10, s[92:95], m0
// GFX89: s_buffer_store_dword s10, s[92:95], m0 ; encoding: [0xae,0x02,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dword tba_lo, s[92:95], m0
// VI: s_buffer_store_dword tba_lo, s[92:95], m0 ; encoding: [0x2e,0x1b,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

s_buffer_store_dword tba_hi, s[92:95], m0
// VI: s_buffer_store_dword tba_hi, s[92:95], m0 ; encoding: [0x6e,0x1b,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

s_buffer_store_dword tma_lo, s[92:95], m0
// VI: s_buffer_store_dword tma_lo, s[92:95], m0 ; encoding: [0xae,0x1b,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

s_buffer_store_dword tma_hi, s[92:95], m0
// VI: s_buffer_store_dword tma_hi, s[92:95], m0 ; encoding: [0xee,0x1b,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOGFX9: error: not a valid operand.

s_buffer_store_dword ttmp0, s[92:95], m0
// VI:   s_buffer_store_dword ttmp0, s[92:95], m0 ; encoding: [0x2e,0x1c,0x60,0xc0,0x7c,0x00,0x00,0x00]
// GFX9: s_buffer_store_dword ttmp0, s[92:95], m0 ; encoding: [0x2e,0x1b,0x60,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[10:11], s[92:95], m0
// GFX89: s_buffer_store_dwordx2 s[10:11], s[92:95], m0 ; encoding: [0xae,0x02,0x64,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[8:11], s[92:95], m0 glc
// GFX89: s_buffer_store_dwordx4 s[8:11], s[92:95], m0 glc ; encoding: [0x2e,0x02,0x69,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: invalid operand for instruction

s_buffer_store_dwordx2 tba, s[92:95], m0 glc
// VI: s_buffer_store_dwordx2 tba, s[92:95], m0 glc ; encoding: [0x2e,0x1b,0x65,0xc0,0x7c,0x00,0x00,0x00]
// NOSICI: error: invalid operand for instruction
// NOGFX9: error: not a valid operand.

s_buffer_load_dword s10, s[92:95], m0
// GFX89: s_buffer_load_dword s10, s[92:95], m0 ; encoding: [0xae,0x02,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword s10, s[92:95], m0 ; encoding: [0x7c,0x5c,0x05,0xc2]

s_buffer_load_dword tba_lo, s[92:95], m0
// VI: s_buffer_load_dword tba_lo, s[92:95], m0 ; encoding: [0x2e,0x1b,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword tba_lo, s[92:95], m0 ; encoding: [0x7c,0x5c,0x36,0xc2]
// NOGFX9: error: not a valid operand.

s_buffer_load_dword tba_hi, s[92:95], m0
// VI: s_buffer_load_dword tba_hi, s[92:95], m0 ; encoding: [0x6e,0x1b,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword tba_hi, s[92:95], m0 ; encoding: [0x7c,0xdc,0x36,0xc2]
// NOGFX9: error: not a valid operand.

s_buffer_load_dword tma_lo, s[92:95], m0
// VI: s_buffer_load_dword tma_lo, s[92:95], m0 ; encoding: [0xae,0x1b,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword tma_lo, s[92:95], m0 ; encoding: [0x7c,0x5c,0x37,0xc2]
// NOGFX9: error: not a valid operand.

s_buffer_load_dword tma_hi, s[92:95], m0
// VI: s_buffer_load_dword tma_hi, s[92:95], m0 ; encoding: [0xee,0x1b,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword tma_hi, s[92:95], m0 ; encoding: [0x7c,0xdc,0x37,0xc2]
// NOGFX9: error: not a valid operand.

s_buffer_load_dword ttmp0, s[92:95], m0
// VI:   s_buffer_load_dword ttmp0, s[92:95], m0 ; encoding: [0x2e,0x1c,0x20,0xc0,0x7c,0x00,0x00,0x00]
// GFX9: s_buffer_load_dword ttmp0, s[92:95], m0 ; encoding: [0x2e,0x1b,0x20,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dword ttmp0, s[92:95], m0 ; encoding: [0x7c,0x5c,0x38,0xc2]

s_buffer_load_dwordx2 s[10:11], s[92:95], m0
// GFX89: s_buffer_load_dwordx2 s[10:11], s[92:95], m0 ; encoding: [0xae,0x02,0x24,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dwordx2 s[10:11], s[92:95], m0 ; encoding: [0x7c,0x5c,0x45,0xc2]

s_buffer_load_dwordx2 tba, s[92:95], m0
// VI:   s_buffer_load_dwordx2 tba, s[92:95], m0 ; encoding: [0x2e,0x1b,0x24,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dwordx2 tba, s[92:95], m0 ; encoding: [0x7c,0x5c,0x76,0xc2]
// NOGFX9: error: not a valid operand.

s_buffer_load_dwordx2 tma, s[92:95], m0
// VI: s_buffer_load_dwordx2 tma, s[92:95], m0 ; encoding: [0xae,0x1b,0x24,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dwordx2 tma, s[92:95], m0 ; encoding: [0x7c,0x5c,0x77,0xc2]
// NOGFX9: error: not a valid operand.

s_buffer_load_dwordx2 ttmp[0:1], s[92:95], m0
// VI:   s_buffer_load_dwordx2 ttmp[0:1], s[92:95], m0 ; encoding: [0x2e,0x1c,0x24,0xc0,0x7c,0x00,0x00,0x00]
// GFX9: s_buffer_load_dwordx2 ttmp[0:1], s[92:95], m0 ; encoding: [0x2e,0x1b,0x24,0xc0,0x7c,0x00,0x00,0x00]
// SICI: s_buffer_load_dwordx2 ttmp[0:1], s[92:95], m0 ; encoding: [0x7c,0x5c,0x78,0xc2]

// FIXME: Should error on SI instead of silently ignoring glc
s_buffer_load_dwordx4 s[8:11], s[92:95], m0 glc
// GFX89: s_buffer_load_dwordx4 s[8:11], s[92:95], m0 glc ; encoding: [0x2e,0x02,0x29,0xc0,0x7c,0x00,0x00,0x00]

//===----------------------------------------------------------------------===//
// s_scratch instructions
//===----------------------------------------------------------------------===//

s_scratch_load_dword s5, s[2:3], s101
// GFX9: s_scratch_load_dword s5, s[2:3], s101 ; encoding: [0x41,0x01,0x14,0xc0,0x65,0x00,0x00,0x00]
// NOSICIVI: error: instruction not supported on this GPU

s_scratch_load_dword s5, s[2:3], s0 glc
// GFX9: s_scratch_load_dword s5, s[2:3], s0 glc ; encoding: [0x41,0x01,0x15,0xc0,0x00,0x00,0x00,0x00]
// NOSICIVI: error

s_scratch_load_dwordx2 s[100:101], s[2:3], s0
// GFX9: s_scratch_load_dwordx2 s[100:101], s[2:3], s0 ; encoding: [0x01,0x19,0x18,0xc0,0x00,0x00,0x00,0x00]
// NOSICIVI: error: instruction not supported on this GPU

s_scratch_load_dwordx2 s[10:11], s[2:3], 0x1 glc
// GFX9: s_scratch_load_dwordx2 s[10:11], s[2:3], 0x1 glc ; encoding: [0x81,0x02,0x1b,0xc0,0x01,0x00,0x00,0x00]
// NOSICIVI: error

s_scratch_load_dwordx4 s[20:23], s[4:5], s0
// GFX9: s_scratch_load_dwordx4 s[20:23], s[4:5], s0 ; encoding: [0x02,0x05,0x1c,0xc0,0x00,0x00,0x00,0x00]
// NOSICIVI: error: instruction not supported on this GPU

s_scratch_store_dword s101, s[4:5], s0
// GFX9: s_scratch_store_dword s101, s[4:5], s0 ; encoding: [0x42,0x19,0x54,0xc0,0x00,0x00,0x00,0x00]
// NOSICIVI: error: instruction not supported on this GPU

s_scratch_store_dword s1, s[4:5], 0x123 glc
// GFX9: s_scratch_store_dword s1, s[4:5], 0x123 glc ; encoding: [0x42,0x00,0x57,0xc0,0x23,0x01,0x00,0x00]
// NOSICIVI: error

s_scratch_store_dwordx2 s[2:3], s[4:5], s101 glc
// GFX9: s_scratch_store_dwordx2 s[2:3], s[4:5], s101 glc ; encoding: [0x82,0x00,0x59,0xc0,0x65,0x00,0x00,0x00]
// NOSICIVI: error

s_scratch_store_dwordx4 s[4:7], s[4:5], s0 glc
// GFX9: s_scratch_store_dwordx4 s[4:7], s[4:5], s0 glc ; encoding: [0x02,0x01,0x5d,0xc0,0x00,0x00,0x00,0x00]
// NOSICIVI: error

//===----------------------------------------------------------------------===//
// s_dcache_discard instructions
//===----------------------------------------------------------------------===//

s_dcache_discard s[2:3], s0
// GFX9:     s_dcache_discard s[2:3], s0 ; encoding: [0x01,0x00,0xa0,0xc0,0x00,0x00,0x00,0x00]
// NOSICIVI: error: instruction not supported on this GPU

s_dcache_discard s[2:3], 0x0
// GFX9:     s_dcache_discard s[2:3], 0x0 ; encoding: [0x01,0x00,0xa2,0xc0,0x00,0x00,0x00,0x00]
// NOSICIVI: error: instruction not supported on this GPU

s_dcache_discard_x2 s[2:3], s101
// GFX9:     s_dcache_discard_x2 s[2:3], s101 ; encoding: [0x01,0x00,0xa4,0xc0,0x65,0x00,0x00,0x00]
// NOSICIVI: error: instruction not supported on this GPU

s_dcache_discard_x2 s[2:3], 0x0
// GFX9:     s_dcache_discard_x2 s[2:3], 0x0 ; encoding: [0x01,0x00,0xa6,0xc0,0x00,0x00,0x00,0x00]
// NOSICIVI: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// s_atomic instructions
//===----------------------------------------------------------------------===//

s_atomic_add s5, s[2:3], s101
// GFX9:     s_atomic_add s5, s[2:3], s101 ; encoding: [0x41,0x01,0x08,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_add s5, s[2:3], 0x0
// GFX9:     s_atomic_add s5, s[2:3], 0x0 ; encoding: [0x41,0x01,0x0a,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_add s5, s[2:3], s0 glc
// GFX9:     s_atomic_add s5, s[2:3], s0 glc ; encoding: [0x41,0x01,0x09,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_add_x2 s[10:11], s[2:3], s101
// GFX9:     s_atomic_add_x2 s[10:11], s[2:3], s101 ; encoding: [0x81,0x02,0x88,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_and s5, s[2:3], s101
// GFX9:     s_atomic_and s5, s[2:3], s101 ; encoding: [0x41,0x01,0x20,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_and_x2 s[10:11], s[2:3], 0x0
// GFX9:     s_atomic_and_x2 s[10:11], s[2:3], 0x0 ; encoding: [0x81,0x02,0xa2,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_cmpswap s[10:11], s[2:3], s101
// GFX9:     s_atomic_cmpswap s[10:11], s[2:3], s101 ; encoding: [0x81,0x02,0x04,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_cmpswap s[10:11], s[2:3], 0x0
// GFX9:     s_atomic_cmpswap s[10:11], s[2:3], 0x0 ; encoding: [0x81,0x02,0x06,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_cmpswap s[10:11], s[2:3], s0 glc
// GFX9:     s_atomic_cmpswap s[10:11], s[2:3], s0 glc ; encoding: [0x81,0x02,0x05,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_cmpswap_x2 s[20:23], s[2:3], s101
// GFX9:     s_atomic_cmpswap_x2 s[20:23], s[2:3], s101 ; encoding: [0x01,0x05,0x84,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_cmpswap_x2 s[20:23], s[2:3], 0x0
// GFX9:     s_atomic_cmpswap_x2 s[20:23], s[2:3], 0x0 ; encoding: [0x01,0x05,0x86,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_cmpswap_x2 s[20:23], s[2:3], s0 glc
// GFX9:     s_atomic_cmpswap_x2 s[20:23], s[2:3], s0 glc ; encoding: [0x01,0x05,0x85,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_dec s5, s[2:3], s0 glc
// GFX9:     s_atomic_dec s5, s[2:3], s0 glc ; encoding: [0x41,0x01,0x31,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_dec_x2 s[10:11], s[2:3], s101
// GFX9:     s_atomic_dec_x2 s[10:11], s[2:3], s101 ; encoding: [0x81,0x02,0xb0,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_inc s5, s[2:3], s0 glc
// GFX9:     s_atomic_inc s5, s[2:3], s0 glc ; encoding: [0x41,0x01,0x2d,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_inc_x2 s[10:11], s[2:3], s101
// GFX9:     s_atomic_inc_x2 s[10:11], s[2:3], s101 ; encoding: [0x81,0x02,0xac,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_or s5, s[2:3], 0x0
// GFX9:     s_atomic_or s5, s[2:3], 0x0 ; encoding: [0x41,0x01,0x26,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_or_x2 s[10:11], s[2:3], s0 glc
// GFX9:     s_atomic_or_x2 s[10:11], s[2:3], s0 glc ; encoding: [0x81,0x02,0xa5,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_smax s5, s[2:3], s101
// GFX9:     s_atomic_smax s5, s[2:3], s101 ; encoding: [0x41,0x01,0x18,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_smax_x2 s[10:11], s[2:3], s0 glc
// GFX9:     s_atomic_smax_x2 s[10:11], s[2:3], s0 glc ; encoding: [0x81,0x02,0x99,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_smin s5, s[2:3], s101
// GFX9:     s_atomic_smin s5, s[2:3], s101 ; encoding: [0x41,0x01,0x10,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_smin_x2 s[10:11], s[2:3], s0 glc
// GFX9:     s_atomic_smin_x2 s[10:11], s[2:3], s0 glc ; encoding: [0x81,0x02,0x91,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_sub s5, s[2:3], s101
// GFX9:     s_atomic_sub s5, s[2:3], s101 ; encoding: [0x41,0x01,0x0c,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_sub_x2 s[10:11], s[2:3], s0 glc
// GFX9:     s_atomic_sub_x2 s[10:11], s[2:3], s0 glc ; encoding: [0x81,0x02,0x8d,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_swap s5, s[2:3], s101
// GFX9:     s_atomic_swap s5, s[2:3], s101 ; encoding: [0x41,0x01,0x00,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_swap_x2 s[10:11], s[2:3], s0 glc
// GFX9:     s_atomic_swap_x2 s[10:11], s[2:3], s0 glc ; encoding: [0x81,0x02,0x81,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_umax s5, s[2:3], s0 glc
// GFX9:     s_atomic_umax s5, s[2:3], s0 glc ; encoding: [0x41,0x01,0x1d,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_umax_x2 s[10:11], s[2:3], s101
// GFX9:     s_atomic_umax_x2 s[10:11], s[2:3], s101 ; encoding: [0x81,0x02,0x9c,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_umin s5, s[2:3], s101
// GFX9:     s_atomic_umin s5, s[2:3], s101 ; encoding: [0x41,0x01,0x14,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_umin_x2 s[10:11], s[2:3], s0 glc
// GFX9:     s_atomic_umin_x2 s[10:11], s[2:3], s0 glc ; encoding: [0x81,0x02,0x95,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_xor s5, s[2:3], s101
// GFX9:     s_atomic_xor s5, s[2:3], s101 ; encoding: [0x41,0x01,0x28,0xc2,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_atomic_xor_x2 s[10:11], s[2:3], s0 glc
// GFX9:     s_atomic_xor_x2 s[10:11], s[2:3], s0 glc ; encoding: [0x81,0x02,0xa9,0xc2,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

//===----------------------------------------------------------------------===//
// s_buffer_atomic instructions
//===----------------------------------------------------------------------===//

s_buffer_atomic_add s5, s[4:7], s101
// GFX9:     s_buffer_atomic_add s5, s[4:7], s101 ; encoding: [0x42,0x01,0x08,0xc1,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_add s5, s[4:7], 0x0
// GFX9:     s_buffer_atomic_add s5, s[4:7], 0x0 ; encoding: [0x42,0x01,0x0a,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_add s5, s[4:7], s0 glc
// GFX9:     s_buffer_atomic_add s5, s[4:7], s0 glc ; encoding: [0x42,0x01,0x09,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_add_x2 s[10:11], s[4:7], s0
// GFX9:     s_buffer_atomic_add_x2 s[10:11], s[4:7], s0 ; encoding: [0x82,0x02,0x88,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_and s101, s[4:7], s0
// GFX9:     s_buffer_atomic_and s101, s[4:7], s0 ; encoding: [0x42,0x19,0x20,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_and_x2 s[10:11], s[8:11], s0
// GFX9:     s_buffer_atomic_and_x2 s[10:11], s[8:11], s0 ; encoding: [0x84,0x02,0xa0,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_cmpswap s[10:11], s[4:7], s0
// GFX9:     s_buffer_atomic_cmpswap s[10:11], s[4:7], s0 ; encoding: [0x82,0x02,0x04,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_cmpswap s[10:11], s[4:7], 0x0
// GFX9:     s_buffer_atomic_cmpswap s[10:11], s[4:7], 0x0 ; encoding: [0x82,0x02,0x06,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_cmpswap s[10:11], s[4:7], s0 glc
// GFX9:     s_buffer_atomic_cmpswap s[10:11], s[4:7], s0 glc ; encoding: [0x82,0x02,0x05,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], s101
// GFX9:     s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], s101 ; encoding: [0x02,0x05,0x84,0xc1,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], 0x0
// GFX9:     s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], 0x0 ; encoding: [0x02,0x05,0x86,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], s0 glc
// GFX9:     s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], s0 glc ; encoding: [0x02,0x05,0x85,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_dec s5, s[4:7], s0
// GFX9:     s_buffer_atomic_dec s5, s[4:7], s0 ; encoding: [0x42,0x01,0x30,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_dec_x2 s[10:11], s[4:7], s0 glc
// GFX9:     s_buffer_atomic_dec_x2 s[10:11], s[4:7], s0 glc ; encoding: [0x82,0x02,0xb1,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_inc s101, s[4:7], s0
// GFX9:     s_buffer_atomic_inc s101, s[4:7], s0 ; encoding: [0x42,0x19,0x2c,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_inc_x2 s[10:11], s[4:7], 0x0
// GFX9:     s_buffer_atomic_inc_x2 s[10:11], s[4:7], 0x0 ; encoding: [0x82,0x02,0xae,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_or s5, s[8:11], s0
// GFX9:     s_buffer_atomic_or s5, s[8:11], s0 ; encoding: [0x44,0x01,0x24,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_or_x2 s[10:11], s[96:99], s0
// GFX9:     s_buffer_atomic_or_x2 s[10:11], s[96:99], s0 ; encoding: [0xb0,0x02,0xa4,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_smax s5, s[4:7], s101
// GFX9:     s_buffer_atomic_smax s5, s[4:7], s101 ; encoding: [0x42,0x01,0x18,0xc1,0x65,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_smax_x2 s[100:101], s[4:7], s0
// GFX9:     s_buffer_atomic_smax_x2 s[100:101], s[4:7], s0 ; encoding: [0x02,0x19,0x98,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_smin s5, s[4:7], 0x0
// GFX9:     s_buffer_atomic_smin s5, s[4:7], 0x0 ; encoding: [0x42,0x01,0x12,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_smin_x2 s[12:13], s[4:7], s0
// GFX9:     s_buffer_atomic_smin_x2 s[12:13], s[4:7], s0 ; encoding: [0x02,0x03,0x90,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_sub s5, s[4:7], s0 glc
// GFX9:     s_buffer_atomic_sub s5, s[4:7], s0 glc ; encoding: [0x42,0x01,0x0d,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_sub_x2 s[10:11], s[4:7], s0
// GFX9:     s_buffer_atomic_sub_x2 s[10:11], s[4:7], s0 ; encoding: [0x82,0x02,0x8c,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_swap s5, s[4:7], s0
// GFX9:     s_buffer_atomic_swap s5, s[4:7], s0 ; encoding: [0x42,0x01,0x00,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_swap_x2 s[10:11], s[4:7], s0 glc
// GFX9:     s_buffer_atomic_swap_x2 s[10:11], s[4:7], s0 glc ; encoding: [0x82,0x02,0x81,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_umax s5, s[4:7], s0
// GFX9:     s_buffer_atomic_umax s5, s[4:7], s0 ; encoding: [0x42,0x01,0x1c,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_umax_x2 s[10:11], s[4:7], s0 glc
// GFX9:     s_buffer_atomic_umax_x2 s[10:11], s[4:7], s0 glc ; encoding: [0x82,0x02,0x9d,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_umin s5, s[4:7], s0
// GFX9:     s_buffer_atomic_umin s5, s[4:7], s0 ; encoding: [0x42,0x01,0x14,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_umin_x2 s[10:11], s[4:7], s0 glc
// GFX9:     s_buffer_atomic_umin_x2 s[10:11], s[4:7], s0 glc ; encoding: [0x82,0x02,0x95,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_xor s5, s[4:7], s0
// GFX9:     s_buffer_atomic_xor s5, s[4:7], s0 ; encoding: [0x42,0x01,0x28,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:

s_buffer_atomic_xor_x2 s[10:11], s[4:7], s0 glc
// GFX9:     s_buffer_atomic_xor_x2 s[10:11], s[4:7], s0 glc ; encoding: [0x82,0x02,0xa9,0xc1,0x00,0x00,0x00,0x00]
// NOSICIVI: error:
