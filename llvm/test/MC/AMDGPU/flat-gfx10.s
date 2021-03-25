// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck --check-prefix=GFX10 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefix=GFX10-ERR --implicit-check-not=error: %s

flat_load_dword v1, v[3:4]
// GFX10: encoding: [0x00,0x00,0x30,0xdc,0x03,0x00,0x7d,0x01]

flat_load_dword v1, v[3:4] offset:-1
// GFX10-ERR: :28: error: expected a 11-bit unsigned offset

flat_load_dword v1, v[3:4] offset:2047
// GFX10: encoding: [0xff,0x07,0x30,0xdc,0x03,0x00,0x7d,0x01]

flat_load_dword v1, v[3:4] offset:2048
// GFX10-ERR: error: expected a 11-bit unsigned offset

flat_load_dword v1, v[3:4] offset:4 glc
// GFX10: encoding: [0x04,0x00,0x31,0xdc,0x03,0x00,0x7d,0x01]

flat_load_dword v1, v[3:4] offset:4 glc slc
// GFX10: encoding: [0x04,0x00,0x33,0xdc,0x03,0x00,0x7d,0x01]

flat_load_dword v1, v[3:4] offset:4 glc slc dlc
// GFX10: encoding: [0x04,0x10,0x33,0xdc,0x03,0x00,0x7d,0x01]

flat_atomic_add v[3:4], v5 offset:8 slc
// GFX10: encoding: [0x08,0x00,0xca,0xdc,0x03,0x05,0x7d,0x00]

flat_atomic_cmpswap v[1:2], v[3:4] offset:2047
// GFX10: encoding: [0xff,0x07,0xc4,0xdc,0x01,0x03,0x7d,0x00]

flat_atomic_cmpswap v[1:2], v[3:4] offset:2047 slc
// GFX10: encoding: [0xff,0x07,0xc6,0xdc,0x01,0x03,0x7d,0x00]

flat_atomic_cmpswap v[1:2], v[3:4]
// GFX10: encoding: [0x00,0x00,0xc4,0xdc,0x01,0x03,0x7d,0x00]

flat_atomic_cmpswap v[1:2], v[3:4] slc
// GFX10: encoding: [0x00,0x00,0xc6,0xdc,0x01,0x03,0x7d,0x00]

flat_atomic_cmpswap v[1:2], v[3:4] offset:2047 glc
// GFX10-ERR: error: instruction must not use glc

flat_atomic_cmpswap v[1:2], v[3:4] glc
// GFX10-ERR: error: instruction must not use glc

flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:2047 glc
// GFX10: encoding: [0xff,0x07,0xc5,0xdc,0x01,0x03,0x7d,0x00]

flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:2047 glc slc
// GFX10: encoding: [0xff,0x07,0xc7,0xdc,0x01,0x03,0x7d,0x00]

flat_atomic_cmpswap v0, v[1:2], v[3:4] glc
// GFX10: encoding: [0x00,0x00,0xc5,0xdc,0x01,0x03,0x7d,0x00]

flat_atomic_cmpswap v0, v[1:2], v[3:4] glc slc
// GFX10: encoding: [0x00,0x00,0xc7,0xdc,0x01,0x03,0x7d,0x00]

flat_atomic_cmpswap v0, v[1:2], v[3:4] glc
// GFX10: encoding: [0x00,0x00,0xc5,0xdc,0x01,0x03,0x7d,0x00]

flat_atomic_cmpswap v0, v[1:2], v[3:4] offset:2047
// GFX10-ERR: error: instruction must use glc

flat_atomic_cmpswap v0, v[1:2], v[3:4] slc
// GFX10-ERR: error: instruction must use glc

flat_atomic_swap v[3:4], v5 offset:16
// GFX10: encoding: [0x10,0x00,0xc0,0xdc,0x03,0x05,0x7d,0x00]

flat_store_dword v[3:4], v1 offset:16
// GFX10: encoding: [0x10,0x00,0x70,0xdc,0x03,0x01,0x7d,0x00]

flat_store_dword v[3:4], v1, off
// GFX10-ERR: error: invalid operand for instruction

flat_store_dword v[3:4], v1, s[0:1]
// GFX10-ERR: error: invalid operand for instruction

flat_store_dword v[3:4], v1, s0
// GFX10-ERR: error: invalid operand for instruction

flat_load_dword v1, v[3:4], off
// GFX10-ERR: error: invalid operand for instruction

flat_load_dword v1, v[3:4], s[0:1]
// GFX10-ERR: error: invalid operand for instruction

flat_load_dword v1, v[3:4], s0
// GFX10-ERR: error: invalid operand for instruction

flat_load_dword v1, v[3:4], exec_hi
// GFX10-ERR: error: invalid operand for instruction

flat_store_dword v[3:4], v1, exec_hi
// GFX10-ERR: error: invalid operand for instruction

flat_load_ubyte_d16 v1, v[3:4]
// GFX10: encoding: [0x00,0x00,0x80,0xdc,0x03,0x00,0x7d,0x01]

flat_load_ubyte_d16_hi v1, v[3:4]
// GFX10: encoding: [0x00,0x00,0x84,0xdc,0x03,0x00,0x7d,0x01]

flat_load_sbyte_d16 v1, v[3:4]
// GFX10: encoding: [0x00,0x00,0x88,0xdc,0x03,0x00,0x7d,0x01]

flat_load_sbyte_d16_hi v1, v[3:4]
// GFX10: encoding: [0x00,0x00,0x8c,0xdc,0x03,0x00,0x7d,0x01]

flat_load_short_d16 v1, v[3:4]
// GFX10: encoding: [0x00,0x00,0x90,0xdc,0x03,0x00,0x7d,0x01]

flat_load_short_d16_hi v1, v[3:4]
// GFX10: encoding: [0x00,0x00,0x94,0xdc,0x03,0x00,0x7d,0x01]

flat_store_byte_d16_hi v[3:4], v1
// GFX10: encoding: [0x00,0x00,0x64,0xdc,0x03,0x01,0x7d,0x00]

flat_store_short_d16_hi v[3:4], v1
// GFX10: encoding: [0x00,0x00,0x6c,0xdc,0x03,0x01,0x7d,0x00]
