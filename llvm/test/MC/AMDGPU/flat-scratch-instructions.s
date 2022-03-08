// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 2>&1 %s | FileCheck -check-prefix=GFX9-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga 2>&1 %s | FileCheck -check-prefix=VI-ERR --implicit-check-not=error: %s

// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck --check-prefix=GFX10 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefix=GFX10-ERR --implicit-check-not=error: %s

scratch_load_ubyte v1, v2, off
// GFX10: encoding: [0x00,0x40,0x20,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_ubyte v1, v2, off      ; encoding: [0x00,0x40,0x40,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_ubyte v1, v2, off dlc
// GFX10: encoding: [0x00,0x50,0x20,0xdc,0x02,0x00,0x7d,0x01]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_load_sbyte v1, v2, off
// GFX10: encoding: [0x00,0x40,0x24,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_sbyte v1, v2, off      ; encoding: [0x00,0x40,0x44,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_sbyte v1, v2, off dlc
// GFX10: encoding: [0x00,0x50,0x24,0xdc,0x02,0x00,0x7d,0x01]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_load_ushort v1, v2, off
// GFX10: encoding: [0x00,0x40,0x28,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_ushort v1, v2, off      ; encoding: [0x00,0x40,0x48,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_ushort v1, v2, off dlc
// GFX10: encoding: [0x00,0x50,0x28,0xdc,0x02,0x00,0x7d,0x01]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_load_sshort v1, v2, off
// GFX10: encoding: [0x00,0x40,0x2c,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_sshort v1, v2, off      ; encoding: [0x00,0x40,0x4c,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_sshort v1, v2, off dlc
// GFX10: encoding: [0x00,0x50,0x2c,0xdc,0x02,0x00,0x7d,0x01]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, v2, off
// GFX10: encoding: [0x00,0x40,0x30,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_dword v1, v2, off ; encoding: [0x00,0x40,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, v2, off dlc
// GFX10: encoding: [0x00,0x50,0x30,0xdc,0x02,0x00,0x7d,0x01]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dwordx2 v[1:2], v3, off
// GFX10: encoding: [0x00,0x40,0x34,0xdc,0x03,0x00,0x7d,0x01]
// GFX9: scratch_load_dwordx2 v[1:2], v3, off      ; encoding: [0x00,0x40,0x54,0xdc,0x03,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dwordx2 v[1:2], v3, off dlc
// GFX10: encoding: [0x00,0x50,0x34,0xdc,0x03,0x00,0x7d,0x01]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dwordx3 v[1:3], v4, off
// GFX10: encoding: [0x00,0x40,0x3c,0xdc,0x04,0x00,0x7d,0x01]
// GFX9: scratch_load_dwordx3 v[1:3], v4, off      ; encoding: [0x00,0x40,0x58,0xdc,0x04,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dwordx3 v[1:3], v4, off dlc
// GFX10: encoding: [0x00,0x50,0x3c,0xdc,0x04,0x00,0x7d,0x01]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dwordx4 v[1:4], v5, off
// GFX10: encoding: [0x00,0x40,0x38,0xdc,0x05,0x00,0x7d,0x01]
// GFX9: scratch_load_dwordx4 v[1:4], v5, off      ; encoding: [0x00,0x40,0x5c,0xdc,0x05,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dwordx4 v[1:4], v5, off dlc
// GFX10: encoding: [0x00,0x50,0x38,0xdc,0x05,0x00,0x7d,0x01]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, v2, off offset:0
// GFX10: encoding: [0x00,0x40,0x30,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_dword v1, v2, off      ; encoding: [0x00,0x40,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, v2, off offset:4095
// GFX10-ERR: :32: error: expected a 12-bit signed offset
// GFX9: scratch_load_dword v1, v2, off offset:4095 ; encoding: [0xff,0x4f,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: :1: error: instruction not supported on this GPU

scratch_load_dword v1, v2, off offset:-1
// GFX10: encoding: [0xff,0x4f,0x30,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_dword v1, v2, off offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, v2, off offset:-4096
// GFX10-ERR: :32: error: expected a 12-bit signed offset
// GFX9: scratch_load_dword v1, v2, off offset:-4096 ; encoding: [0x00,0x50,0x50,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: :1: error: instruction not supported on this GPU

scratch_load_dword v1, v2, off offset:4096
// GFX10-ERR: :32: error: expected a 12-bit signed offset
// GFX9-ERR: :32: error: expected a 13-bit signed offset
// VI-ERR: :1: error: instruction not supported on this GPU

scratch_load_dword v1, v2, off offset:-4097
// GFX10-ERR: :32: error: expected a 12-bit signed offset
// GFX9-ERR: :32: error: expected a 13-bit signed offset
// VI-ERR: :1: error: instruction not supported on this GPU

scratch_load_dword v0, v1, off offset:-2049 glc slc
// GFX10-ERR: :32: error: expected a 12-bit signed offset
// GFX9: scratch_load_dword v0, v1, off offset:-2049 glc slc ; encoding: [0xff,0x57,0x53,0xdc,0x01,0x00,0x7f,0x00]
// VI-ERR: :1: error: instruction not supported on this GPU

scratch_load_dword v0, v1, off offset:-2048 glc slc
// GFX10: scratch_load_dword v0, v1, off offset:-2048 glc slc ; encoding: [0x00,0x48,0x33,0xdc,0x01,0x00,0x7d,0x00]
// GFX9: scratch_load_dword v0, v1, off offset:-2048 glc slc ; encoding: [0x00,0x58,0x53,0xdc,0x01,0x00,0x7f,0x00]
// VI-ERR: :1: error: instruction not supported on this GPU

scratch_load_dword v255, off, s1 offset:2047
// GFX10: scratch_load_dword v255, off, s1 offset:2047 ; encoding: [0xff,0x47,0x30,0xdc,0x00,0x00,0x01,0xff]
// GFX9: scratch_load_dword v255, off, s1 offset:2047 ; encoding: [0xff,0x47,0x50,0xdc,0x00,0x00,0x01,0xff]
// VI-ERR: :1: error: instruction not supported on this GPU

scratch_load_dword v255, off, s0 offset:2048
// GFX10-ERR: :34: error: expected a 12-bit signed offset
// GFX9: scratch_load_dword v255, off, s0 offset:2048 ; encoding: [0x00,0x48,0x50,0xdc,0x00,0x00,0x00,0xff]
// VI-ERR: :1: error: instruction not supported on this GPU

scratch_store_byte v1, v2, off
// GFX10: encoding: [0x00,0x40,0x60,0xdc,0x01,0x02,0x7d,0x00]
// GFX9: scratch_store_byte v1, v2, off ; encoding: [0x00,0x40,0x60,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_byte v1, v2, off dlc
// GFX10: encoding: [0x00,0x50,0x60,0xdc,0x01,0x02,0x7d,0x00]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_store_short v1, v2, off
// GFX10: encoding: [0x00,0x40,0x68,0xdc,0x01,0x02,0x7d,0x00]
// GFX9: scratch_store_short v1, v2, off ; encoding: [0x00,0x40,0x68,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_short v1, v2, off dlc
// GFX10: encoding: [0x00,0x50,0x68,0xdc,0x01,0x02,0x7d,0x00]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword v1, v2, off
// GFX10: encoding: [0x00,0x40,0x70,0xdc,0x01,0x02,0x7d,0x00]
// GFX9: scratch_store_dword v1, v2, off ; encoding: [0x00,0x40,0x70,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword v1, v2, off dlc
// GFX10: encoding: [0x00,0x50,0x70,0xdc,0x01,0x02,0x7d,0x00]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dwordx2 v1, v[2:3], off
// GFX10: encoding: [0x00,0x40,0x74,0xdc,0x01,0x02,0x7d,0x00]
// GFX9: scratch_store_dwordx2 v1, v[2:3], off ; encoding: [0x00,0x40,0x74,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dwordx2 v1, v[2:3], off dlc
// GFX10: encoding: [0x00,0x50,0x74,0xdc,0x01,0x02,0x7d,0x00]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dwordx3 v1, v[2:4], off
// GFX10: encoding: [0x00,0x40,0x7c,0xdc,0x01,0x02,0x7d,0x00]
// GFX9: scratch_store_dwordx3 v1, v[2:4], off ; encoding: [0x00,0x40,0x78,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dwordx3 v1, v[2:4], off dlc
// GFX10: encoding: [0x00,0x50,0x7c,0xdc,0x01,0x02,0x7d,0x00]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dwordx4 v1, v[2:5], off
// GFX10: encoding: [0x00,0x40,0x78,0xdc,0x01,0x02,0x7d,0x00]
// GFX9: scratch_store_dwordx4 v1, v[2:5], off ; encoding: [0x00,0x40,0x7c,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dwordx4 v1, v[2:5], off dlc
// GFX10: encoding: [0x00,0x50,0x78,0xdc,0x01,0x02,0x7d,0x00]
// GFX9-ERR: error: dlc modifier is not supported on this GPU
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword v1, v2, off offset:12
// GFX10: encoding: [0x0c,0x40,0x70,0xdc,0x01,0x02,0x7d,0x00]
// GFX9: scratch_store_dword v1, v2, off offset:12 ; encoding: [0x0c,0x40,0x70,0xdc,0x01,0x02,0x7f,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, off, s1
// GFX10: encoding: [0x00,0x40,0x30,0xdc,0x00,0x00,0x01,0x01]
// GFX9: scratch_load_dword v1, off, s1 ; encoding: [0x00,0x40,0x50,0xdc,0x00,0x00,0x01,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, off, s1 offset:32
// GFX10: encoding: [0x20,0x40,0x30,0xdc,0x00,0x00,0x01,0x01]
// GFX9: scratch_load_dword v1, off, s1 offset:32 ; encoding: [0x20,0x40,0x50,0xdc,0x00,0x00,0x01,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword off, v2, s1
// GFX10: encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x01,0x00]
// GFX9: scratch_store_dword off, v2, s1 ; encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x01,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword off, v2, s1 offset:12
// GFX10: encoding: [0x0c,0x40,0x70,0xdc,0x00,0x02,0x01,0x00]
// GFX9: scratch_store_dword off, v2, s1 offset:12 ; encoding: [0x0c,0x40,0x70,0xdc,0x00,0x02,0x01,0x00]
// VI-ERR: error: instruction not supported on this GPU

// FIXME: Should error about multiple offsets
scratch_load_dword v1, v2, s1
// GFX10-ERR: error: operands are not valid for this GPU or mode
// GFX9-ERR: error: operands are not valid for this GPU or mode
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, v2, s1 offset:32
// GFX10-ERR: error: operands are not valid for this GPU or mode
// GFX9-ERR: error: operands are not valid for this GPU or mode
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword v1, v2, s1
// GFX10-ERR: error: operands are not valid for this GPU or mode
// GFX9-ERR: error: operands are not valid for this GPU or mode
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword v1, v2, s1 offset:32
// GFX10-ERR: error: operands are not valid for this GPU or mode
// GFX9-ERR: error: operands are not valid for this GPU or mode
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, off, exec_hi
// GFX10-ERR: error: invalid operand for instruction
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword off, v2, exec_hi
// GFX10-ERR: error: invalid operand for instruction
// GFX9-ERR: error: invalid operand for instruction
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, off, exec_lo
// GFX10: encoding: [0x00,0x40,0x30,0xdc,0x00,0x00,0x7e,0x01]
// GFX9: scratch_load_dword v1, off, exec_lo ; encoding: [0x00,0x40,0x50,0xdc,0x00,0x00,0x7e,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword off, v2, exec_lo
// GFX10: encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x7e,0x00]
// GFX9: scratch_store_dword off, v2, exec_lo ; encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x7e,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_dword v1, off, m0
// GFX10: encoding: [0x00,0x40,0x30,0xdc,0x00,0x00,0x7c,0x01]
// GFX9: scratch_load_dword v1, off, m0  ; encoding: [0x00,0x40,0x50,0xdc,0x00,0x00,0x7c,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_dword off, v2, m0
// GFX10: encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x7c,0x00]
// GFX9: scratch_store_dword off, v2, m0 ; encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x7c,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_ubyte_d16 v1, v2, off
// GFX10: encoding: [0x00,0x40,0x80,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_ubyte_d16 v1, v2, off ; encoding: [0x00,0x40,0x80,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_ubyte_d16_hi v1, v2, off
// GFX10: encoding: [0x00,0x40,0x84,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_ubyte_d16_hi v1, v2, off ; encoding: [0x00,0x40,0x84,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_sbyte_d16 v1, v2, off
// GFX10: encoding: [0x00,0x40,0x88,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_sbyte_d16 v1, v2, off ; encoding: [0x00,0x40,0x88,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_sbyte_d16_hi v1, v2, off
// GFX10: encoding: [0x00,0x40,0x8c,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_sbyte_d16_hi v1, v2, off ; encoding: [0x00,0x40,0x8c,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_short_d16 v1, v2, off
// GFX10: encoding: [0x00,0x40,0x90,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_short_d16 v1, v2, off ; encoding: [0x00,0x40,0x90,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_load_short_d16_hi v1, v2, off
// GFX10: encoding: [0x00,0x40,0x94,0xdc,0x02,0x00,0x7d,0x01]
// GFX9: scratch_load_short_d16_hi v1, v2, off ; encoding: [0x00,0x40,0x94,0xdc,0x02,0x00,0x7f,0x01]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_byte_d16_hi off, v2, s1
// GFX10: encoding: [0x00,0x40,0x64,0xdc,0x00,0x02,0x01,0x00]
// GFX9: scratch_store_byte_d16_hi off, v2, s1 ; encoding: [0x00,0x40,0x64,0xdc,0x00,0x02,0x01,0x00]
// VI-ERR: error: instruction not supported on this GPU

scratch_store_short_d16_hi off, v2, s1
// GFX10: encoding: [0x00,0x40,0x6c,0xdc,0x00,0x02,0x01,0x00]
// GFX9: scratch_store_short_d16_hi off, v2, s1 ; encoding: [0x00,0x40,0x6c,0xdc,0x00,0x02,0x01,0x00]
// VI-ERR: error: instruction not supported on this GPU
