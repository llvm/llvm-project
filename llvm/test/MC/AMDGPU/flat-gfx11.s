// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga 2>&1 %s | FileCheck -check-prefix=VI-GFX9_10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 2>&1 %s | FileCheck -check-prefix=VI-GFX9_10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 2>&1 %s | FileCheck --check-prefix=VI-GFX9_10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1030 2>&1 %s | FileCheck --check-prefix=VI-GFX9_10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s


// FLAT


flat_load_u8 v1, v[3:4]
// GFX11-LABEL: flat_load_u8 v1, v[3:4] ; encoding: [0x00,0x00,0x40,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_i8 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x44,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_u16 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x48,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_i16 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x4c,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_b16 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x80,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x50,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:-1
// GFX11-ERR: :28: error: expected an 11-bit unsigned offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:2047
// GFX11: encoding: [0xff,0x07,0x50,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:2048
// GFX11-ERR: error: expected an 11-bit unsigned offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:4 glc
// GFX11: encoding: [0x04,0x40,0x50,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:4 glc slc
// GFX11: encoding: [0x04,0xc0,0x50,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:4 glc slc dlc
// GFX11: encoding: [0x04,0xe0,0x50,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b64 v[1:2], v[3:4]
// GFX11: encoding: [0x00,0x00,0x54,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b96 v[1:3], v[5:6]
// GFX11: encoding: [0x00,0x00,0x58,0xdc,0x05,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b128 v[1:4], v[5:6]
// GFX11: encoding: [0x00,0x00,0x5c,0xdc,0x05,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_i8 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x7c,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_hi_i8 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x88,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b8 v[3:4], v1
// GFX11: encoding: [0x00,0x00,0x60,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b16 v[3:4], v1
// GFX11: encoding: [0x00,0x00,0x64,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1 offset:16
// GFX11: encoding: [0x10,0x00,0x68,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1, off
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1, s[0:1]
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1, s0
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4], off
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4], s[0:1]
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4], s0
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4], exec_hi
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1, exec_hi
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b64 v[1:2], v[3:4]
// GFX11: encoding: [0x00,0x00,0x6c,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b96 v[1:2], v[3:5]
// GFX11: encoding: [0x00,0x00,0x70,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b128 v[1:2], v[3:6]
// GFX11: encoding: [0x00,0x00,0x74,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 offset:2047
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v[1:2], v3 offset:2047 glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v[1:2], v3 glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 offset:2047 glc
// GFX11: encoding: [0xff,0x47,0xcc,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0xcc,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 glc
// GFX11: encoding: [0x00,0x40,0xcc,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 glc slc
// GFX11: encoding: [0x00,0xc0,0xcc,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 offset:2047 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4] offset:2047 glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4] glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4], v[5:6] offset:2047 glc
// GFX11: encoding: [0xff,0x47,0x04,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4], v[5:6] offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0x04,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4], v[5:6] glc
// GFX11: encoding: [0x00,0x40,0x04,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4], v[5:6] glc slc
// GFX11: encoding: [0x00,0xc0,0x04,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_add_u32 v[3:4], v5 slc
// GFX11: encoding: [0x00,0x80,0xd4,0xdc,0x03,0x05,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_add_u32 v2, v[3:4], v5 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_add_u32 v1, v[3:4], v5 offset:8 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] offset:2047
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v[1:2], v[3:4] offset:2047 glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v[1:2], v[3:4] glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] offset:2047 glc
// GFX11: encoding: [0xff,0x47,0xd0,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0xd0,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] glc
// GFX11: encoding: [0x00,0x40,0xd0,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] glc slc
// GFX11: encoding: [0x00,0xc0,0xd0,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4]
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] offset:2047 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4] offset:2047
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4] offset:2047 glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4] glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8] offset:2047 glc
// GFX11: encoding: [0xff,0x47,0x08,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8] offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0x08,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8] glc
// GFX11: encoding: [0x00,0x40,0x08,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8] glc slc
// GFX11: encoding: [0x00,0xc0,0x08,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_u8 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x78,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_hi_u8 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x84,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_i8 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x7c,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_hi_i8 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x88,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_b16 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x80,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_hi_b16 v1, v[3:4]
// GFX11: encoding: [0x00,0x00,0x8c,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_d16_hi_b8 v[3:4], v1
// GFX11: encoding: [0x00,0x00,0x90,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_d16_hi_b16 v[3:4], v1
// GFX11: encoding: [0x00,0x00,0x94,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU


// GLOBAL No saddr


global_load_u8 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x42,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_i8 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x46,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_u16 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x4a,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_i16 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x4e,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_b16 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x82,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x52,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:-1
// GFX11-ERR: :28: error: expected an 11-bit unsigned offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:2047
// GFX11: encoding: [0xff,0x07,0x52,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:2048
// GFX11-ERR: error: expected an 11-bit unsigned offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:4 glc
// GFX11: encoding: [0x04,0x40,0x52,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:4 glc slc
// GFX11: encoding: [0x04,0xc0,0x52,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:4 glc slc dlc
// GFX11: encoding: [0x04,0xe0,0x52,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b64 v[1:2], v[3:4], off
// GFX11: encoding: [0x00,0x00,0x56,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b96 v[1:3], v[5:6], off
// GFX11: encoding: [0x00,0x00,0x5a,0xdc,0x05,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b128 v[1:4], v[5:6], off
// GFX11: encoding: [0x00,0x00,0x5e,0xdc,0x05,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_i8 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x7e,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_i8 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x8a,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b8 v[3:4], v1, off
// GFX11: encoding: [0x00,0x00,0x62,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b16 v[3:4], v1, off
// GFX11: encoding: [0x00,0x00,0x66,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1, off offset:16
// GFX11: encoding: [0x10,0x00,0x6a,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1, s[0:1]
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1, s0
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off, s[0:1]
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], s0
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], exec_hi
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1, exec_hi
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b64 v[1:2], v[3:4], off
// GFX11: encoding: [0x00,0x00,0x6e,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b96 v[1:2], v[3:5], off
// GFX11: encoding: [0x00,0x00,0x72,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b128 v[1:2], v[3:6], off
// GFX11: encoding: [0x00,0x00,0x76,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off offset:2047
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v[1:2], v3 offset:2047 glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v[1:2], v3 glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off offset:2047 glc
// GFX11: encoding: [0xff,0x47,0xce,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0xce,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off glc
// GFX11: encoding: [0x00,0x40,0xce,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off glc slc
// GFX11: encoding: [0x00,0xc0,0xce,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off offset:2047 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v[3:4], v[5:6], off offset:2047 glc
// GFX11: encoding: [0xff,0x47,0x06,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v[3:4], v[5:6], off offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0x06,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v[3:4], v[5:6], off glc
// GFX11: encoding: [0x00,0x40,0x06,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v[3:4], v[5:6], off glc slc
// GFX11: encoding: [0x00,0xc0,0x06,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_add_u32 v2, v[3:4], off, v5 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_add_u32 v1, v[3:4], off, v5 offset:8 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off offset:2047
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off offset:2047 glc
// GFX11: encoding: [0xff,0x47,0xd2,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0xd2,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off glc
// GFX11: encoding: [0x00,0x40,0xd2,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off glc slc
// GFX11: encoding: [0x00,0xc0,0xd2,0xdc,0x01,0x03,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], off, v[3:4]
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off offset:2047 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v[3:4], off offset:2047
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// TODO-GFX11 FIXME global_atomic_cmpswap_b64, also GFX10?
global_atomic_cmpswap_b64 v[1:4], v3, v[5:8], off offset:2047 glc
// GFX11-NOT: encoding: [0xff,0x47,0x0a,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// TODO-GFX11 FIXME global_atomic_cmpswap_b64, also GFX10?
global_atomic_cmpswap_b64 v[1:4], v3, v[5:8], off offset:2047 glc slc
// GFX11-NOT: encoding: [0xff,0xc7,0x0a,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// TODO-GFX11 FIXME global_atomic_cmpswap_b64, also GFX10?
global_atomic_cmpswap_b64 v[1:4], v3, v[5:8], off glc
// GFX11-NOT: encoding: [0x00,0x40,0x0a,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// TODO-GFX11 FIXME global_atomic_cmpswap_b64, also GFX10?
global_atomic_cmpswap_b64 v[1:4], v3, v[5:8], off glc slc
// GFX11-NOT: encoding: [0x00,0xc0,0x0a,0xdd,0x03,0x05,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_u8 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x7a,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_u8 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x86,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_i8 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x7e,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_i8 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x8a,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_b16 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x82,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_b16 v1, v[3:4], off
// GFX11: encoding: [0x00,0x00,0x8e,0xdc,0x03,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_d16_hi_b8 v[3:4], v1, off
// GFX11: encoding: [0x00,0x00,0x92,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_d16_hi_b16 v[3:4], v1, off
// GFX11: encoding: [0x00,0x00,0x96,0xdc,0x03,0x01,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_addtid_b32 v1, off
// GFX11: encoding: [0x00,0x00,0xa2,0xdc,0x00,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU


// GLOBAL With saddr


global_load_u8 v1, v3, s2
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_u8 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x42,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_i8 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x46,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_u16 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x4a,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_i16 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x4e,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_b16 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x82,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x52,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:-1
// GFX11-ERR: :28: error: expected an 11-bit unsigned offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:2047
// GFX11: encoding: [0xff,0x07,0x52,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:2048
// GFX11-ERR: error: expected an 11-bit unsigned offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:4 glc
// GFX11: encoding: [0x04,0x40,0x52,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:4 glc slc
// GFX11: encoding: [0x04,0xc0,0x52,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:4 glc slc dlc
// GFX11: encoding: [0x04,0xe0,0x52,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b64 v[1:2], v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x56,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b96 v[1:3], v5, s[2:3]
// GFX11: encoding: [0x00,0x00,0x5a,0xdc,0x05,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b128 v[1:4], v5, s[2:3]
// GFX11: encoding: [0x00,0x00,0x5e,0xdc,0x05,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_i8 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x7e,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_i8 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x8a,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b8 v3, v1, s[2:3]
// GFX11: encoding: [0x00,0x00,0x62,0xdc,0x03,0x01,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b16 v3, v1, s[2:3]
// GFX11: encoding: [0x00,0x00,0x66,0xdc,0x03,0x01,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1, s[2:3] offset:16
// GFX11: encoding: [0x10,0x00,0x6a,0xdc,0x03,0x01,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1, s[0:1]
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1, s0
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3]
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3], s[0:1]
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s0
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, exec_hi
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1, exec_hi
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b64 v1, v[2:3], s[2:3]
// GFX11: encoding: [0x00,0x00,0x6e,0xdc,0x01,0x02,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b96 v1, v[3:5], s[2:3]
// GFX11: encoding: [0x00,0x00,0x72,0xdc,0x01,0x03,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b128 v1, v[3:6], s[2:3]
// GFX11: encoding: [0x00,0x00,0x76,0xdc,0x01,0x03,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, s[2:3] offset:2047
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v[1:2], v3 offset:2047 glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v[1:2], v3 glc
// GFX11-ERR: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] offset:2047 glc
// GFX11: encoding: [0xff,0x47,0xce,0xdc,0x01,0x03,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0xce,0xdc,0x01,0x03,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] glc
// GFX11: encoding: [0x00,0x40,0xce,0xdc,0x01,0x03,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] glc slc
// GFX11: encoding: [0x00,0xc0,0xce,0xdc,0x01,0x03,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3]
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, s[2:3] offset:2047 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v3, v[5:6], s[2:3] offset:2047 glc
// GFX11: encoding: [0xff,0x47,0x06,0xdd,0x03,0x05,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v3, v[5:6], s[2:3] offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0x06,0xdd,0x03,0x05,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v3, v[5:6], s[2:3] glc
// GFX11: encoding: [0x00,0x40,0x06,0xdd,0x03,0x05,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v3, v[5:6], s[2:3] glc slc
// GFX11: encoding: [0x00,0xc0,0x06,0xdd,0x03,0x05,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_add_u32 v2, v3, s[2:3], v5 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_add_u32 v1, v3, s[2:3], v5 offset:8 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v3, s[2:3] offset:2047
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v[2:3], s[2:3] offset:2047 glc
// GFX11: encoding: [0xff,0x47,0xd2,0xdc,0x01,0x02,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v[2:3], s[2:3] offset:2047 glc slc
// GFX11: encoding: [0xff,0xc7,0xd2,0xdc,0x01,0x02,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v[2:3], s[2:3] glc
// GFX11: encoding: [0x00,0x40,0xd2,0xdc,0x01,0x02,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v[2:3], s[2:3] glc slc
// GFX11: encoding: [0x00,0xc0,0xd2,0xdc,0x01,0x02,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, s[2:3], v3
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v3, s[2:3] slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v3, s[2:3] offset:2047 slc
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v3, s[2:3] offset:2047
// GFX11-ERR: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// TODO-GFX11 FIXME global_atomic_cmpswap_b64, also GFX10?
global_atomic_cmpswap_b64 v[1:4], v3, v[5:8], s[2:3] offset:2047 glc
// GFX11-NOT: encoding: [0xff,0x47,0x0a,0xdd,0x03,0x05,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// TODO-GFX11 FIXME global_atomic_cmpswap_b64, also GFX10?
global_atomic_cmpswap_b64 v[1:4], v3, v[5:8], s[2:3] offset:2047 glc slc
// GFX11-NOT: encoding: [0xff,0xc7,0x0a,0xdd,0x03,0x05,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// TODO-GFX11 FIXME global_atomic_cmpswap_b64, also GFX10?
global_atomic_cmpswap_b64 v[1:4], v3, v[5:8], s[2:3] glc
// GFX11-NOT: encoding: [0x00,0x40,0x0a,0xdd,0x03,0x05,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// TODO-GFX11 FIXME global_atomic_cmpswap_b64, also GFX10?
global_atomic_cmpswap_b64 v[1:4], v3, v[5:8], s[2:3] glc slc
// GFX11-NOT: encoding: [0x00,0xc0,0x0a,0xdd,0x03,0x05,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_u8 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x7a,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_u8 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x86,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_i8 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x7e,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_i8 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x8a,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_b16 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x82,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_b16 v1, v3, s[2:3]
// GFX11: encoding: [0x00,0x00,0x8e,0xdc,0x03,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_d16_hi_b8 v3, v1, s[2:3]
// GFX11: encoding: [0x00,0x00,0x92,0xdc,0x03,0x01,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_d16_hi_b16 v3, v1, s[2:3]
// GFX11: encoding: [0x00,0x00,0x96,0xdc,0x03,0x01,0x02,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_addtid_b32 v1, s[2:3]
// GFX11: encoding: [0x00,0x00,0xa2,0xdc,0x00,0x00,0x02,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU


// SCRATCH


scratch_load_u8 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x41,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_i8 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x45,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_u16 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x49,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_i16 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x4d,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x51,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b64 v[1:2], v2, s1
// GFX11: encoding: [0x00,0x00,0x55,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b96 v[1:3], v2, s1
// GFX11: encoding: [0x00,0x00,0x59,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b128 v[1:4], v2, s1
// GFX11: encoding: [0x00,0x00,0x5d,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b8 v1, v2, s3
// GFX11: encoding: [0x00,0x00,0x61,0xdc,0x01,0x02,0x83,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b16 v1, v2, s3
// GFX11: encoding: [0x00,0x00,0x65,0xdc,0x01,0x02,0x83,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s3
// GFX11: encoding: [0x00,0x00,0x69,0xdc,0x01,0x02,0x83,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b64 v1, v[2:3], s3
// GFX11: encoding: [0x00,0x00,0x6d,0xdc,0x01,0x02,0x83,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b96 v1, v[2:4], s3
// GFX11: encoding: [0x00,0x00,0x71,0xdc,0x01,0x02,0x83,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b128 v1, v[2:5], s3
// GFX11: encoding: [0x00,0x00,0x75,0xdc,0x01,0x02,0x83,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_u8 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x79,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_hi_u8 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x85,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_i8 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x7d,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_hi_i8 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x89,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_b16 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x81,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_hi_b16 v1, v2, s1
// GFX11: encoding: [0x00,0x00,0x8d,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_d16_hi_b8 v1, v2, s3
// GFX11: encoding: [0x00,0x00,0x91,0xdc,0x01,0x02,0x83,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_d16_hi_b16 v1, v2, s3
// GFX11: encoding: [0x00,0x00,0x95,0xdc,0x01,0x02,0x83,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:2047
// GFX11: encoding: [0xff,0x07,0x51,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, off offset:2047
// GFX11: encoding: [0xff,0x07,0x51,0xdc,0x02,0x00,0xfc,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, off, s1 offset:2047
// GFX11: encoding: [0xff,0x07,0x51,0xdc,0x00,0x00,0x01,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, off, off offset:2047
// GFX11: encoding: [0xff,0x07,0x51,0xdc,0x00,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, off, off
// GFX11: encoding: [0x00,0x00,0x51,0xdc,0x00,0x00,0x7c,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s3 offset:2047
// GFX11: encoding: [0xff,0x07,0x69,0xdc,0x01,0x02,0x83,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, off offset:2047
// GFX11: encoding: [0xff,0x07,0x69,0xdc,0x01,0x02,0xfc,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 off, v2, s3 offset:2047
// GFX11: encoding: [0xff,0x07,0x69,0xdc,0x00,0x02,0x03,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 off, v2, off offset:2047
// GFX11: encoding: [0xff,0x07,0x69,0xdc,0x00,0x02,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:4095
// GFX11: encoding: [0xff,0x0f,0x51,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:-4096
// GFX11: encoding: [0x00,0x10,0x51,0xdc,0x02,0x00,0x81,0x01]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s1 offset:4095
// GFX11: encoding: [0xff,0x0f,0x69,0xdc,0x01,0x02,0x81,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s1 offset:-4096
// GFX11: encoding: [0x00,0x10,0x69,0xdc,0x01,0x02,0x81,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:4096
// GFX11-ERR: error: expected an 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:-4097
// GFX11-ERR: error: expected an 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s1 offset:4096
// GFX11-ERR: error: expected an 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s1 offset:-4097
// GFX11-ERR: error: expected an 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 off, v2, off
// GFX11: encoding: [0x00,0x00,0x69,0xdc,0x00,0x02,0x7c,0x00]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU
