// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 -check-prefix=GCN %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding 2>&1 %s | FileCheck -check-prefix=GFX9-ERR -check-prefix=GCNERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding 2>&1 %s | FileCheck -check-prefix=VI-ERR -check-prefix=GCNERR %s

scratch_load_ubyte v1, v2, off
// GFX9: scratch_load_ubyte v1, v2, off      ; encoding: [0x00,0x40,0x40,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_sbyte v1, v2, off
// GFX9: scratch_load_sbyte v1, v2, off      ; encoding: [0x00,0x40,0x44,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_ushort v1, v2, off
// GFX9: scratch_load_ushort v1, v2, off      ; encoding: [0x00,0x40,0x48,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_sshort v1, v2, off
// GFX9: scratch_load_sshort v1, v2, off      ; encoding: [0x00,0x40,0x4c,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_dword v1, v2, off
// GFX9: scratch_load_dword v1, v2, off ; encoding: [0x00,0x40,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_dwordx2 v[1:2], v3, off
// GFX9: scratch_load_dwordx2 v[1:2], v3, off      ; encoding: [0x00,0x40,0x54,0xdc,0x03,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_dwordx3 v[1:3], v4, off
// GFX9: scratch_load_dwordx3 v[1:3], v4, off      ; encoding: [0x00,0x40,0x58,0xdc,0x04,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_dwordx4 v[1:4], v5, off
// GFX9: scratch_load_dwordx4 v[1:4], v5, off      ; encoding: [0x00,0x40,0x5c,0xdc,0x05,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU
// FIXME: VI error should be instruction nto supported

scratch_load_dword v1, v2, off offset:0
// GFX9: scratch_load_dword v1, v2, off      ; encoding: [0x00,0x40,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: not a valid operand.

scratch_load_dword v1, v2, off offset:4095
// GFX9: scratch_load_dword v1, v2, off offset:4095 ; encoding: [0xff,0x4f,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: not a valid operand.

scratch_load_dword v1, v2, off offset:-1
// GFX9: scratch_load_dword v1, v2, off offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: not a valid operand.

scratch_load_dword v1, v2, off offset:-4096
// GFX9: scratch_load_dword v1, v2, off offset:-4096 ; encoding: [0x00,0x50,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: not a valid operand.

scratch_load_dword v1, v2, off offset:4096
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: not a valid operand.

scratch_load_dword v1, v2, off offset:-4097
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: not a valid operand.

scratch_store_byte v1, v2, off
// GFX9: scratch_store_byte v1, v2, off ; encoding: [0x00,0x40,0x60,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_store_short v1, v2, off
// GFX9: scratch_store_short v1, v2, off ; encoding: [0x00,0x40,0x68,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_store_dword v1, v2, off
// GFX9: scratch_store_dword v1, v2, off ; encoding: [0x00,0x40,0x70,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_store_dwordx2 v1, v[2:3], off
// GFX9: scratch_store_dwordx2 v1, v[2:3], off ; encoding: [0x00,0x40,0x74,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_store_dwordx3 v1, v[2:4], off
// GFX9: scratch_store_dwordx3 v1, v[2:4], off ; encoding: [0x00,0x40,0x78,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_store_dwordx4 v1, v[2:5], off
// GFX9: scratch_store_dwordx4 v1, v[2:5], off ; encoding: [0x00,0x40,0x7c,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_store_dword v1, v2, off offset:12
// GFX9: scratch_store_dword v1, v2, off offset:12 ; encoding: [0x0c,0x40,0x70,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: error: not a valid operand

scratch_load_dword v1, off, s1
// GFX9: scratch_load_dword v1, off, s1 ; encoding: [0x00,0x40,0x50,0xdc,0x00,0x00,0x01,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_dword v1, off, s1 offset:32
// GFX9: scratch_load_dword v1, off, s1 offset:32 ; encoding: [0x20,0x40,0x50,0xdc,0x00,0x00,0x01,0x01]
// VI-ERR: error: not a valid operand

scratch_store_dword off, v2, s1
// GFX9: scratch_store_dword off, v2, s1 ; encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x01,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_store_dword off, v2, s1 offset:12
// GFX9: scratch_store_dword off, v2, s1 offset:12 ; encoding: [0x0c,0x40,0x70,0xdc,0x00,0x02,0x01,0x00]
// VI-ERR: error: not a valid operand

// FIXME: Should error about multiple offsets
scratch_load_dword v1, v2, s1
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: invalid operand for instruction

scratch_load_dword v1, v2, s1 offset:32
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: not a valid operand

scratch_store_dword v1, v2, s1
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: invalid operand for instruction

scratch_store_dword v1, v2, s1 offset:32
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: not a valid operand

scratch_load_dword v1, off, exec_hi
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: invalid operand for instruction

scratch_store_dword off, v2, exec_hi
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: invalid operand for instruction

scratch_load_dword v1, off, exec_lo
// GFX9: scratch_load_dword v1, off, exec_lo ; encoding: [0x00,0x40,0x50,0xdc,0x00,0x00,0x7e,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_store_dword off, v2, exec_lo
// GFX9: scratch_store_dword off, v2, exec_lo ; encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x7e,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_load_dword v1, off, m0
// GFX9: scratch_load_dword v1, off, m0  ; encoding: [0x00,0x40,0x50,0xdc,0x00,0x00,0x7c,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_store_dword off, v2, m0
// GFX9: scratch_store_dword off, v2, m0 ; encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x7c,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_load_ubyte_d16 v1, v2, off
// GFX9: scratch_load_ubyte_d16 v1, v2, off ; encoding: [0x00,0x40,0x80,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_ubyte_d16_hi v1, v2, off
// GFX9: scratch_load_ubyte_d16_hi v1, v2, off ; encoding: [0x00,0x40,0x84,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_sbyte_d16 v1, v2, off
// GFX9: scratch_load_sbyte_d16 v1, v2, off ; encoding: [0x00,0x40,0x88,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_sbyte_d16_hi v1, v2, off
// GFX9: scratch_load_sbyte_d16_hi v1, v2, off ; encoding: [0x00,0x40,0x8c,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_short_d16 v1, v2, off
// GFX9: scratch_load_short_d16 v1, v2, off ; encoding: [0x00,0x40,0x90,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_load_short_d16_hi v1, v2, off
// GFX9: scratch_load_short_d16_hi v1, v2, off ; encoding: [0x00,0x40,0x94,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: instruction not supported on this GPU

scratch_store_byte_d16_hi off, v2, s1
// GFX9: scratch_store_byte_d16_hi off, v2, s1 ; encoding: [0x00,0x40,0x64,0xdc,0x00,0x02,0x01,0x00]
// VI-ERR: instruction not supported on this GPU

scratch_store_short_d16_hi off, v2, s1
// GFX9: scratch_store_short_d16_hi off, v2, s1 ; encoding: [0x00,0x40,0x6c,0xdc,0x00,0x02,0x01,0x00]
// VI-ERR: instruction not supported on this GPU
