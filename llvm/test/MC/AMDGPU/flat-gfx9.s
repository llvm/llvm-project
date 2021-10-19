// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 -check-prefix=GCN %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=VI -check-prefix=GCN %s

// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 2>&1 %s | FileCheck -check-prefix=GFX9-ERR -check-prefix=GCNERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga 2>&1 %s | FileCheck -check-prefix=VI-ERR -check-prefix=GCNERR --implicit-check-not=error: %s


flat_load_dword v1, v[3:4] offset:0
// GCN: flat_load_dword v1, v[3:4]      ; encoding: [0x00,0x00,0x50,0xdc,0x03,0x00,0x00,0x01]

flat_load_dword v1, v[3:4] offset:-1
// VI-ERR: :28: error: flat offset modifier is not supported on this GPU
// GFX9-ERR: :28: error: expected a 12-bit unsigned offset

flat_load_dword v1, v[3:4] offset:4095
// GFX9: flat_load_dword v1, v[3:4] offset:4095 ; encoding: [0xff,0x0f,0x50,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: :28: error: flat offset modifier is not supported on this GPU

flat_load_dword v1, v[3:4] offset:4096
// VI-ERR: :28: error: flat offset modifier is not supported on this GPU
// GFX9-ERR: :28: error: expected a 12-bit unsigned offset

flat_load_dword v1, v[3:4] offset:4 glc
// GFX9: flat_load_dword v1, v[3:4] offset:4 glc ; encoding: [0x04,0x00,0x51,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: :28: error: flat offset modifier is not supported on this GPU

flat_load_dword v1, v[3:4] offset:4 glc slc
// GFX9: flat_load_dword v1, v[3:4] offset:4 glc slc ; encoding: [0x04,0x00,0x53,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: :28: error: flat offset modifier is not supported on this GPU

flat_atomic_add v[3:4], v5 offset:8 slc
// GFX9: flat_atomic_add v[3:4], v5 offset:8 slc ; encoding: [0x08,0x00,0x0a,0xdd,0x03,0x05,0x00,0x00]
// VI-ERR: :28: error: flat offset modifier is not supported on this GPU

flat_atomic_add v[3:4], v5 inst_offset:8 slc
// GFX9: flat_atomic_add v[3:4], v5 offset:8 slc ; encoding: [0x08,0x00,0x0a,0xdd,0x03,0x05,0x00,0x00]
// VI-ERR: :28: error: flat offset modifier is not supported on this GPU

flat_atomic_cmpswap v[1:2], v[3:4] offset:4095
// GFX9: flat_atomic_cmpswap v[1:2], v[3:4] offset:4095 ; encoding: [0xff,0x0f,0x04,0xdd,0x01,0x03,0x00,0x00]
// VI-ERR: :36: error: flat offset modifier is not supported on this GPU

flat_atomic_cmpswap v[1:2], v[3:4] offset:4095 slc
// GFX9: flat_atomic_cmpswap v[1:2], v[3:4] offset:4095 slc ; encoding: [0xff,0x0f,0x06,0xdd,0x01,0x03,0x00,0x00]
// VI-ERR: :36: error: flat offset modifier is not supported on this GPU

flat_atomic_cmpswap v[1:2], v[3:4]
// GFX9: flat_atomic_cmpswap v[1:2], v[3:4] ; encoding: [0x00,0x00,0x04,0xdd,0x01,0x03,0x00,0x00]
// VI:   flat_atomic_cmpswap v[1:2], v[3:4] ; encoding: [0x00,0x00,0x04,0xdd,0x01,0x03,0x00,0x00]

flat_atomic_cmpswap v[1:2], v[3:4] slc
// GFX9: flat_atomic_cmpswap v[1:2], v[3:4] slc ; encoding: [0x00,0x00,0x06,0xdd,0x01,0x03,0x00,0x00]
// VI:   flat_atomic_cmpswap v[1:2], v[3:4] slc ; encoding: [0x00,0x00,0x06,0xdd,0x01,0x03,0x00,0x00]

flat_atomic_cmpswap v[1:2], v[3:4] offset:4095 glc
// GFX9-ERR: error: instruction must not use glc
// VI-ERR: error: flat offset modifier is not supported on this GPU

flat_atomic_cmpswap v[1:2], v[3:4] glc
// GCNERR: error: instruction must not use glc

flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:4095 glc
// GFX9: flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:4095 glc ; encoding: [0xff,0x0f,0x05,0xdd,0x01,0x03,0x00,0x00]
// VI-ERR: :40: error: flat offset modifier is not supported on this GPU

flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:4095 glc slc
// GFX9: flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:4095 glc slc ; encoding: [0xff,0x0f,0x07,0xdd,0x01,0x03,0x00,0x00]
// VI-ERR: :40: error: flat offset modifier is not supported on this GPU

flat_atomic_cmpswap v0, v[1:2], v[3:4] glc
// GFX9: flat_atomic_cmpswap v0, v[1:2], v[3:4] glc ; encoding: [0x00,0x00,0x05,0xdd,0x01,0x03,0x00,0x00]
// VI:   flat_atomic_cmpswap v0, v[1:2], v[3:4] glc ; encoding: [0x00,0x00,0x05,0xdd,0x01,0x03,0x00,0x00]

flat_atomic_cmpswap v0, v[1:2], v[3:4] glc slc
// GFX9: flat_atomic_cmpswap v0, v[1:2], v[3:4] glc slc ; encoding: [0x00,0x00,0x07,0xdd,0x01,0x03,0x00,0x00]
// VI:   flat_atomic_cmpswap v0, v[1:2], v[3:4] glc slc ; encoding: [0x00,0x00,0x07,0xdd,0x01,0x03,0x00,0x00]

flat_atomic_cmpswap v0, v[1:2], v[3:4]
// GCNERR: error: instruction must use glc

flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:4095
// GFX9-ERR: error: instruction must use glc
// VI-ERR: error: flat offset modifier is not supported on this GPU

flat_atomic_cmpswap v0, v[1:2], v[3:4] slc
// GCNERR: error: instruction must use glc

flat_atomic_swap v[3:4], v5 offset:16
// GFX9: flat_atomic_swap v[3:4], v5 offset:16 ; encoding: [0x10,0x00,0x00,0xdd,0x03,0x05,0x00,0x00]
// VI-ERR: :29: error: flat offset modifier is not supported on this GPU

flat_store_dword v[3:4], v1 offset:16
// GFX9: flat_store_dword v[3:4], v1 offset:16 ; encoding: [0x10,0x00,0x70,0xdc,0x03,0x01,0x00,0x00]
// VI-ERR: :29: error: flat offset modifier is not supported on this GPU

flat_store_dword v[3:4], v1, off
// GCNERR: :30: error: invalid operand for instruction

flat_store_dword v[3:4], v1, s[0:1]
// GCNERR: :30: error: invalid operand for instruction

flat_store_dword v[3:4], v1, s0
// GCNERR: :30: error: invalid operand for instruction

flat_load_dword v1, v[3:4], off
// GCNERR: :29: error: invalid operand for instruction

flat_load_dword v1, v[3:4], s[0:1]
// GCNERR: :29: error: invalid operand for instruction

flat_load_dword v1, v[3:4], s0
// GCNERR: :29: error: invalid operand for instruction

flat_load_dword v1, v[3:4], exec_hi
// GCNERR: :29: error: invalid operand for instruction

flat_store_dword v[3:4], v1, exec_hi
// GCNERR: :30: error: invalid operand for instruction

flat_load_ubyte_d16 v1, v[3:4]
// GFX9: flat_load_ubyte_d16 v1, v[3:4]  ; encoding: [0x00,0x00,0x80,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: error: instruction not supported on this GPU

flat_load_ubyte_d16_hi v1, v[3:4]
// GFX9: flat_load_ubyte_d16_hi v1, v[3:4] ; encoding: [0x00,0x00,0x84,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: error: instruction not supported on this GPU

flat_load_sbyte_d16 v1, v[3:4]
// GFX9: flat_load_sbyte_d16 v1, v[3:4]  ; encoding: [0x00,0x00,0x88,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: error: instruction not supported on this GPU

flat_load_sbyte_d16_hi v1, v[3:4]
// GFX9: flat_load_sbyte_d16_hi v1, v[3:4] ; encoding: [0x00,0x00,0x8c,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: error: instruction not supported on this GPU

flat_load_short_d16 v1, v[3:4]
// GFX9: flat_load_short_d16 v1, v[3:4]  ; encoding: [0x00,0x00,0x90,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: error: instruction not supported on this GPU

flat_load_short_d16_hi v1, v[3:4]
// GFX9: flat_load_short_d16_hi v1, v[3:4] ; encoding: [0x00,0x00,0x94,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: error: instruction not supported on this GPU

flat_store_byte_d16_hi v[3:4], v1
// GFX9: flat_store_byte_d16_hi v[3:4], v1 ; encoding: [0x00,0x00,0x64,0xdc,0x03,0x01,0x00,0x00]
// VI-ERR: error: instruction not supported on this GPU

flat_store_short_d16_hi v[3:4], v1
// GFX9: flat_store_short_d16_hi v[3:4], v1 ; encoding: [0x00,0x00,0x6c,0xdc,0x03,0x01,0x00,0x00
// VI-ERR: error: instruction not supported on this GPU
