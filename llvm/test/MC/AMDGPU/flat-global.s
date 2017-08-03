// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 -check-prefix=GCN %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding 2>&1 %s | FileCheck -check-prefix=GFX9-ERR -check-prefix=GCNERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding 2>&1 %s | FileCheck -check-prefix=VI-ERR -check-prefix=GCNERR %s

global_load_ubyte v1, v[3:4]
// GFX9: global_load_ubyte v1, v[3:4]      ; encoding: [0x00,0x80,0x40,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: instruction not supported on this GPU

global_load_sbyte v1, v[3:4]
// GFX9: global_load_sbyte v1, v[3:4]      ; encoding: [0x00,0x80,0x44,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: instruction not supported on this GPU

global_load_ushort v1, v[3:4]
// GFX9: global_load_ushort v1, v[3:4]      ; encoding: [0x00,0x80,0x48,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: instruction not supported on this GPU

global_load_sshort v1, v[3:4]
// GFX9: global_load_sshort v1, v[3:4]      ; encoding: [0x00,0x80,0x4c,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: instruction not supported on this GPU

global_load_dword v1, v[3:4]
// GFX9: global_load_dword v1, v[3:4]      ; encoding: [0x00,0x80,0x50,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: instruction not supported on this GPU

global_load_dwordx2 v[1:2], v[3:4]
// GFX9: global_load_dwordx2 v[1:2], v[3:4]      ; encoding: [0x00,0x80,0x54,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: instruction not supported on this GPU

global_load_dwordx3 v[1:3], v[3:4]
// GFX9: global_load_dwordx3 v[1:3], v[3:4]      ; encoding: [0x00,0x80,0x58,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: instruction not supported on this GPU

global_load_dwordx4 v[1:4], v[3:4]
// GFX9: global_load_dwordx4 v[1:4], v[3:4]      ; encoding: [0x00,0x80,0x5c,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: instruction not supported on this GPU
// FIXME: VI error should be instruction nto supported
global_load_dword v1, v[3:4] offset:0
// GFX9: global_load_dword v1, v[3:4]      ; encoding: [0x00,0x80,0x50,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: :36: error: not a valid operand.

global_load_dword v1, v[3:4] offset:4095
// GFX9: global_load_dword v1, v[3:4] offset:4095 ; encoding: [0xff,0x8f,0x50,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: :36: error: not a valid operand.

global_load_dword v1, v[3:4] offset:-1
// GFX9: global_load_dword v1, v[3:4] offset:-1 ; encoding: [0xff,0x9f,0x50,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: :36: error: not a valid operand.

global_load_dword v1, v[3:4] offset:-4096
// GFX9: global_load_dword v1, v[3:4] offset:-4096 ; encoding: [0x00,0x90,0x50,0xdc,0x03,0x00,0x00,0x01]
// VI-ERR: :36: error: not a valid operand.

global_load_dword v1, v[3:4] offset:4096
// GFX9-ERR: :30: error: invalid operand for instruction
// VI-ERR: :36: error: not a valid operand.

global_load_dword v1, v[3:4] offset:-4097
// GFX9-ERR: :30: error: invalid operand for instruction
// VI-ERR: :36: error: not a valid operand.

global_store_byte v[3:4], v1
// GFX9: global_store_byte v[3:4], v1 ; encoding: [0x00,0x80,0x60,0xdc,0x03,0x01,0x00,0x00]
// VI-ERR: instruction not supported on this GPU

global_store_short v[3:4], v1
// GFX9: global_store_short v[3:4], v1 ; encoding: [0x00,0x80,0x68,0xdc,0x03,0x01,0x00,0x00]
// VI-ERR: instruction not supported on this GPU

global_store_dword v[3:4], v1
// GFX9: global_store_dword v[3:4], v1 ; encoding: [0x00,0x80,0x70,0xdc,0x03,0x01,0x00,0x00]
// VI-ERR: instruction not supported on this GPU

global_store_dwordx2 v[3:4], v[1:2]
// GFX9: global_store_dwordx2 v[3:4], v[1:2] ; encoding: [0x00,0x80,0x74,0xdc,0x03,0x01,0x00,0x00]
// VI-ERR: instruction not supported on this GPU

global_store_dwordx3 v[3:4], v[1:3]
// GFX9: global_store_dwordx3 v[3:4], v[1:3] ; encoding: [0x00,0x80,0x78,0xdc,0x03,0x01,0x00,0x00]
// VI-ERR: instruction not supported on this GPU

global_store_dwordx4 v[3:4], v[1:4]
// GFX9: global_store_dwordx4 v[3:4], v[1:4] ; encoding: [0x00,0x80,0x7c,0xdc,0x03,0x01,0x00,0x00]
// VI-ERR: instruction not supported on this GPU

global_store_dword v[3:4], v1 offset:12
// GFX9: global_store_dword v[3:4], v1 offset:12 ; encoding: [0x0c,0x80,0x70,0xdc,0x03,0x01,0x00,0x00]
// VI-ERR: :37: error: not a valid operand
