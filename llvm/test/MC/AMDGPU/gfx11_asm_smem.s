// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefixes=GFX11 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefixes=GFX11-ERR --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// ENC_SMEM.
//===----------------------------------------------------------------------===//

s_load_b32 s5, s[2:3], s0
// GFX11: encoding: [0x41,0x01,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 s101, s[2:3], s0
// GFX11: encoding: [0x41,0x19,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 vcc_lo, s[2:3], s0
// GFX11: encoding: [0x81,0x1a,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 vcc_hi, s[2:3], s0
// GFX11: encoding: [0xc1,0x1a,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 s5, s[4:5], s0
// GFX11: encoding: [0x42,0x01,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 s5, s[100:101], s0
// GFX11: encoding: [0x72,0x01,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 s5, vcc, s0
// GFX11: encoding: [0x75,0x01,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 s5, s[2:3], s101
// GFX11: encoding: [0x41,0x01,0x00,0xf4,0x00,0x00,0x00,0xca]

s_load_b32 s5, s[2:3], vcc_lo
// GFX11: encoding: [0x41,0x01,0x00,0xf4,0x00,0x00,0x00,0xd4]

s_load_b32 s5, s[2:3], vcc_hi
// GFX11: encoding: [0x41,0x01,0x00,0xf4,0x00,0x00,0x00,0xd6]

s_load_b32 s5, s[2:3], m0
// GFX11: encoding: [0x41,0x01,0x00,0xf4,0x00,0x00,0x00,0xfa]

s_load_b32 s5, s[2:3], 0x0
// GFX11: encoding: [0x41,0x01,0x00,0xf4,0x00,0x00,0x00,0xf8]

s_load_b32 s5, s[2:3], s7 offset:0x12345
// GFX11: encoding: [0x41,0x01,0x00,0xf4,0x45,0x23,0x01,0x0e]

s_load_b32 s5, s[2:3], s0 glc
// GFX11: encoding: [0x41,0x41,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 s5, s[2:3], s0 dlc
// GFX11: encoding: [0x41,0x21,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 s5, s[2:3], s0 glc dlc
// GFX11: encoding: [0x41,0x61,0x00,0xf4,0x00,0x00,0x00,0x00]

s_load_b32 s5, s[2:3], 0x1234 glc dlc
// GFX11: encoding: [0x41,0x61,0x00,0xf4,0x34,0x12,0x00,0xf8]

s_load_b64 s[10:11], s[2:3], s0
// GFX11: encoding: [0x81,0x02,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 s[12:13], s[2:3], s0
// GFX11: encoding: [0x01,0x03,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 s[100:101], s[2:3], s0
// GFX11: encoding: [0x01,0x19,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 vcc, s[2:3], s0
// GFX11: encoding: [0x81,0x1a,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 s[10:11], s[4:5], s0
// GFX11: encoding: [0x82,0x02,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 s[10:11], s[100:101], s0
// GFX11: encoding: [0xb2,0x02,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 s[10:11], vcc, s0
// GFX11: encoding: [0xb5,0x02,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 s[10:11], s[2:3], s101
// GFX11: encoding: [0x81,0x02,0x04,0xf4,0x00,0x00,0x00,0xca]

s_load_b64 s[10:11], s[2:3], vcc_lo
// GFX11: encoding: [0x81,0x02,0x04,0xf4,0x00,0x00,0x00,0xd4]

s_load_b64 s[10:11], s[2:3], vcc_hi
// GFX11: encoding: [0x81,0x02,0x04,0xf4,0x00,0x00,0x00,0xd6]

s_load_b64 s[10:11], s[2:3], m0
// GFX11: encoding: [0x81,0x02,0x04,0xf4,0x00,0x00,0x00,0xfa]

s_load_b64 s[10:11], s[2:3], 0x0
// GFX11: encoding: [0x81,0x02,0x04,0xf4,0x00,0x00,0x00,0xf8]

s_load_b64 s[10:11], s[2:3], s0 glc
// GFX11: encoding: [0x81,0x42,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 s[10:11], s[2:3], s0 dlc
// GFX11: encoding: [0x81,0x22,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 s[10:11], s[2:3], s0 glc dlc
// GFX11: encoding: [0x81,0x62,0x04,0xf4,0x00,0x00,0x00,0x00]

s_load_b64 s[10:11], s[2:3], 0x1234 glc dlc
// GFX11: encoding: [0x81,0x62,0x04,0xf4,0x34,0x12,0x00,0xf8]

s_load_b128 s[20:23], s[2:3], s0
// GFX11: encoding: [0x01,0x05,0x08,0xf4,0x00,0x00,0x00,0x00]

s_load_b128 s[24:27], s[2:3], s0
// GFX11: encoding: [0x01,0x06,0x08,0xf4,0x00,0x00,0x00,0x00]

s_load_b128 s[96:99], s[2:3], s0
// GFX11: encoding: [0x01,0x18,0x08,0xf4,0x00,0x00,0x00,0x00]

s_load_b128 s[20:23], s[4:5], s0
// GFX11: encoding: [0x02,0x05,0x08,0xf4,0x00,0x00,0x00,0x00]

s_load_b128 s[20:23], s[100:101], s0
// GFX11: encoding: [0x32,0x05,0x08,0xf4,0x00,0x00,0x00,0x00]

s_load_b128 s[20:23], vcc, s0
// GFX11: encoding: [0x35,0x05,0x08,0xf4,0x00,0x00,0x00,0x00]

s_load_b128 s[20:23], s[2:3], s101
// GFX11: encoding: [0x01,0x05,0x08,0xf4,0x00,0x00,0x00,0xca]

s_load_b128 s[20:23], s[2:3], vcc_lo
// GFX11: encoding: [0x01,0x05,0x08,0xf4,0x00,0x00,0x00,0xd4]

s_load_b128 s[20:23], s[2:3], vcc_hi
// GFX11: encoding: [0x01,0x05,0x08,0xf4,0x00,0x00,0x00,0xd6]

s_load_b128 s[20:23], s[2:3], m0
// GFX11: encoding: [0x01,0x05,0x08,0xf4,0x00,0x00,0x00,0xfa]

s_load_b128 s[20:23], s[2:3], 0x0
// GFX11: encoding: [0x01,0x05,0x08,0xf4,0x00,0x00,0x00,0xf8]

s_load_b128 s[20:23], s[2:3], s0 glc
// GFX11: encoding: [0x01,0x45,0x08,0xf4,0x00,0x00,0x00,0x00]

s_load_b128 s[20:23], s[2:3], s0 dlc
// GFX11: encoding: [0x01,0x25,0x08,0xf4,0x00,0x00,0x00,0x00]

s_load_b128 s[20:23], s[2:3], s0 glc dlc
// GFX11: encoding: [0x01,0x65,0x08,0xf4,0x00,0x00,0x00,0x00]

s_load_b128 s[20:23], s[2:3], 0x1234 glc dlc
// GFX11: encoding: [0x01,0x65,0x08,0xf4,0x34,0x12,0x00,0xf8]

s_load_b256 s[20:27], s[2:3], s0
// GFX11: encoding: [0x01,0x05,0x0c,0xf4,0x00,0x00,0x00,0x00]

s_load_b256 s[24:31], s[2:3], s0
// GFX11: encoding: [0x01,0x06,0x0c,0xf4,0x00,0x00,0x00,0x00]

s_load_b256 s[92:99], s[2:3], s0
// GFX11: encoding: [0x01,0x17,0x0c,0xf4,0x00,0x00,0x00,0x00]

s_load_b256 s[20:27], s[4:5], s0
// GFX11: encoding: [0x02,0x05,0x0c,0xf4,0x00,0x00,0x00,0x00]

s_load_b256 s[20:27], s[100:101], s0
// GFX11: encoding: [0x32,0x05,0x0c,0xf4,0x00,0x00,0x00,0x00]

s_load_b256 s[20:27], vcc, s0
// GFX11: encoding: [0x35,0x05,0x0c,0xf4,0x00,0x00,0x00,0x00]

s_load_b256 s[20:27], s[2:3], s101
// GFX11: encoding: [0x01,0x05,0x0c,0xf4,0x00,0x00,0x00,0xca]

s_load_b256 s[20:27], s[2:3], vcc_lo
// GFX11: encoding: [0x01,0x05,0x0c,0xf4,0x00,0x00,0x00,0xd4]

s_load_b256 s[20:27], s[2:3], vcc_hi
// GFX11: encoding: [0x01,0x05,0x0c,0xf4,0x00,0x00,0x00,0xd6]

s_load_b256 s[20:27], s[2:3], m0
// GFX11: encoding: [0x01,0x05,0x0c,0xf4,0x00,0x00,0x00,0xfa]

s_load_b256 s[20:27], s[2:3], 0x0
// GFX11: encoding: [0x01,0x05,0x0c,0xf4,0x00,0x00,0x00,0xf8]

s_load_b256 s[20:27], s[2:3], s0 glc
// GFX11: encoding: [0x01,0x45,0x0c,0xf4,0x00,0x00,0x00,0x00]

s_load_b256 s[20:27], s[2:3], s0 dlc
// GFX11: encoding: [0x01,0x25,0x0c,0xf4,0x00,0x00,0x00,0x00]

s_load_b256 s[20:27], s[2:3], s0 glc dlc
// GFX11: encoding: [0x01,0x65,0x0c,0xf4,0x00,0x00,0x00,0x00]

s_load_b256 s[20:27], s[2:3], 0x1234 glc dlc
// GFX11: encoding: [0x01,0x65,0x0c,0xf4,0x34,0x12,0x00,0xf8]

s_load_b512 s[20:35], s[2:3], s0
// GFX11: encoding: [0x01,0x05,0x10,0xf4,0x00,0x00,0x00,0x00]

s_load_b512 s[24:39], s[2:3], s0
// GFX11: encoding: [0x01,0x06,0x10,0xf4,0x00,0x00,0x00,0x00]

s_load_b512 s[84:99], s[2:3], s0
// GFX11: encoding: [0x01,0x15,0x10,0xf4,0x00,0x00,0x00,0x00]

s_load_b512 s[20:35], s[4:5], s0
// GFX11: encoding: [0x02,0x05,0x10,0xf4,0x00,0x00,0x00,0x00]

s_load_b512 s[20:35], s[100:101], s0
// GFX11: encoding: [0x32,0x05,0x10,0xf4,0x00,0x00,0x00,0x00]

s_load_b512 s[20:35], vcc, s0
// GFX11: encoding: [0x35,0x05,0x10,0xf4,0x00,0x00,0x00,0x00]

s_load_b512 s[20:35], s[2:3], s101
// GFX11: encoding: [0x01,0x05,0x10,0xf4,0x00,0x00,0x00,0xca]

s_load_b512 s[20:35], s[2:3], vcc_lo
// GFX11: encoding: [0x01,0x05,0x10,0xf4,0x00,0x00,0x00,0xd4]

s_load_b512 s[20:35], s[2:3], vcc_hi
// GFX11: encoding: [0x01,0x05,0x10,0xf4,0x00,0x00,0x00,0xd6]

s_load_b512 s[20:35], s[2:3], m0
// GFX11: encoding: [0x01,0x05,0x10,0xf4,0x00,0x00,0x00,0xfa]

s_load_b512 s[20:35], s[2:3], 0x0
// GFX11: encoding: [0x01,0x05,0x10,0xf4,0x00,0x00,0x00,0xf8]

s_load_b512 s[20:35], s[2:3], s0 glc
// GFX11: encoding: [0x01,0x45,0x10,0xf4,0x00,0x00,0x00,0x00]

s_load_b512 s[20:35], s[2:3], s0 dlc
// GFX11: encoding: [0x01,0x25,0x10,0xf4,0x00,0x00,0x00,0x00]

s_load_b512 s[20:35], s[2:3], s0 glc dlc
// GFX11: encoding: [0x01,0x65,0x10,0xf4,0x00,0x00,0x00,0x00]

s_load_b512 s[20:35], s[2:3], 0x1234 glc dlc
// GFX11: encoding: [0x01,0x65,0x10,0xf4,0x34,0x12,0x00,0xf8]

s_buffer_load_b32 s5, s[4:7], s0
// GFX11: encoding: [0x42,0x01,0x20,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b32 s101, s[4:7], s0
// GFX11: encoding: [0x42,0x19,0x20,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b32 vcc_lo, s[4:7], s0
// GFX11: encoding: [0x82,0x1a,0x20,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b32 vcc_hi, s[4:7], s0
// GFX11: encoding: [0xc2,0x1a,0x20,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b32 s5, s[8:11], s0
// GFX11: encoding: [0x44,0x01,0x20,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b32 s5, s[96:99], s0
// GFX11: encoding: [0x70,0x01,0x20,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b32 s5, s[4:7], s101
// GFX11: encoding: [0x42,0x01,0x20,0xf4,0x00,0x00,0x00,0xca]

s_buffer_load_b32 s5, s[4:7], vcc_lo
// GFX11: encoding: [0x42,0x01,0x20,0xf4,0x00,0x00,0x00,0xd4]

s_buffer_load_b32 s5, s[4:7], vcc_hi
// GFX11: encoding: [0x42,0x01,0x20,0xf4,0x00,0x00,0x00,0xd6]

s_buffer_load_b32 s5, s[4:7], m0
// GFX11: encoding: [0x42,0x01,0x20,0xf4,0x00,0x00,0x00,0xfa]

s_buffer_load_b32 s5, s[4:7], 0x0
// GFX11: encoding: [0x42,0x01,0x20,0xf4,0x00,0x00,0x00,0xf8]

s_buffer_load_b32 s5, s[4:7], s0 glc
// GFX11: encoding: [0x42,0x41,0x20,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b32 s5, s[4:7], s0 dlc
// GFX11: encoding: [0x42,0x21,0x20,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b32 s5, s[4:7], s0 glc dlc
// GFX11: encoding: [0x42,0x61,0x20,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b32 s5, s[4:7], 0x1234 glc dlc
// GFX11: encoding: [0x42,0x61,0x20,0xf4,0x34,0x12,0x00,0xf8]

s_buffer_load_b64 s[10:11], s[4:7], s0
// GFX11: encoding: [0x82,0x02,0x24,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b64 s[12:13], s[4:7], s0
// GFX11: encoding: [0x02,0x03,0x24,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b64 s[100:101], s[4:7], s0
// GFX11: encoding: [0x02,0x19,0x24,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b64 vcc, s[4:7], s0
// GFX11: encoding: [0x82,0x1a,0x24,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b64 s[10:11], s[8:11], s0
// GFX11: encoding: [0x84,0x02,0x24,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b64 s[10:11], s[96:99], s0
// GFX11: encoding: [0xb0,0x02,0x24,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b64 s[10:11], s[4:7], s101
// GFX11: encoding: [0x82,0x02,0x24,0xf4,0x00,0x00,0x00,0xca]

s_buffer_load_b64 s[10:11], s[4:7], vcc_lo
// GFX11: encoding: [0x82,0x02,0x24,0xf4,0x00,0x00,0x00,0xd4]

s_buffer_load_b64 s[10:11], s[4:7], vcc_hi
// GFX11: encoding: [0x82,0x02,0x24,0xf4,0x00,0x00,0x00,0xd6]

s_buffer_load_b64 s[10:11], s[4:7], m0
// GFX11: encoding: [0x82,0x02,0x24,0xf4,0x00,0x00,0x00,0xfa]

s_buffer_load_b64 s[10:11], s[4:7], 0x0
// GFX11: encoding: [0x82,0x02,0x24,0xf4,0x00,0x00,0x00,0xf8]

s_buffer_load_b64 s[10:11], s[4:7], s0 glc
// GFX11: encoding: [0x82,0x42,0x24,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b64 s[10:11], s[4:7], s0 dlc
// GFX11: encoding: [0x82,0x22,0x24,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b64 s[10:11], s[4:7], s0 glc dlc
// GFX11: encoding: [0x82,0x62,0x24,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b64 s[10:11], s[4:7], 0x1234 glc dlc
// GFX11: encoding: [0x82,0x62,0x24,0xf4,0x34,0x12,0x00,0xf8]

s_buffer_load_b128 s[20:23], s[4:7], s0
// GFX11: encoding: [0x02,0x05,0x28,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b128 s[24:27], s[4:7], s0
// GFX11: encoding: [0x02,0x06,0x28,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b128 s[96:99], s[4:7], s0
// GFX11: encoding: [0x02,0x18,0x28,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b128 s[20:23], s[8:11], s0
// GFX11: encoding: [0x04,0x05,0x28,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b128 s[20:23], s[96:99], s0
// GFX11: encoding: [0x30,0x05,0x28,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b128 s[20:23], s[4:7], s101
// GFX11: encoding: [0x02,0x05,0x28,0xf4,0x00,0x00,0x00,0xca]

s_buffer_load_b128 s[20:23], s[4:7], vcc_lo
// GFX11: encoding: [0x02,0x05,0x28,0xf4,0x00,0x00,0x00,0xd4]

s_buffer_load_b128 s[20:23], s[4:7], vcc_hi
// GFX11: encoding: [0x02,0x05,0x28,0xf4,0x00,0x00,0x00,0xd6]

s_buffer_load_b128 s[20:23], s[4:7], m0
// GFX11: encoding: [0x02,0x05,0x28,0xf4,0x00,0x00,0x00,0xfa]

s_buffer_load_b128 s[20:23], s[4:7], 0x0
// GFX11: encoding: [0x02,0x05,0x28,0xf4,0x00,0x00,0x00,0xf8]

s_buffer_load_b128 s[20:23], s[4:7], s0 glc
// GFX11: encoding: [0x02,0x45,0x28,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b128 s[20:23], s[4:7], s0 dlc
// GFX11: encoding: [0x02,0x25,0x28,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b128 s[20:23], s[4:7], s0 glc dlc
// GFX11: encoding: [0x02,0x65,0x28,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b128 s[20:23], s[4:7], 0x1234 glc dlc
// GFX11: encoding: [0x02,0x65,0x28,0xf4,0x34,0x12,0x00,0xf8]

s_buffer_load_b256 s[20:27], s[4:7], s0
// GFX11: encoding: [0x02,0x05,0x2c,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b256 s[24:31], s[4:7], s0
// GFX11: encoding: [0x02,0x06,0x2c,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b256 s[92:99], s[4:7], s0
// GFX11: encoding: [0x02,0x17,0x2c,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b256 s[20:27], s[8:11], s0
// GFX11: encoding: [0x04,0x05,0x2c,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b256 s[20:27], s[96:99], s0
// GFX11: encoding: [0x30,0x05,0x2c,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b256 s[20:27], s[4:7], s101
// GFX11: encoding: [0x02,0x05,0x2c,0xf4,0x00,0x00,0x00,0xca]

s_buffer_load_b256 s[20:27], s[4:7], vcc_lo
// GFX11: encoding: [0x02,0x05,0x2c,0xf4,0x00,0x00,0x00,0xd4]

s_buffer_load_b256 s[20:27], s[4:7], vcc_hi
// GFX11: encoding: [0x02,0x05,0x2c,0xf4,0x00,0x00,0x00,0xd6]

s_buffer_load_b256 s[20:27], s[4:7], m0
// GFX11: encoding: [0x02,0x05,0x2c,0xf4,0x00,0x00,0x00,0xfa]

s_buffer_load_b256 s[20:27], s[4:7], 0x0
// GFX11: encoding: [0x02,0x05,0x2c,0xf4,0x00,0x00,0x00,0xf8]

s_buffer_load_b256 s[20:27], s[4:7], s0 glc
// GFX11: encoding: [0x02,0x45,0x2c,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b256 s[20:27], s[4:7], s0 dlc
// GFX11: encoding: [0x02,0x25,0x2c,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b256 s[20:27], s[4:7], s0 glc dlc
// GFX11: encoding: [0x02,0x65,0x2c,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b256 s[20:27], s[4:7], 0x1234 glc dlc
// GFX11: encoding: [0x02,0x65,0x2c,0xf4,0x34,0x12,0x00,0xf8]

s_buffer_load_b512 s[20:35], s[4:7], s0
// GFX11: encoding: [0x02,0x05,0x30,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b512 s[24:39], s[4:7], s0
// GFX11: encoding: [0x02,0x06,0x30,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b512 s[84:99], s[4:7], s0
// GFX11: encoding: [0x02,0x15,0x30,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b512 s[20:35], s[8:11], s0
// GFX11: encoding: [0x04,0x05,0x30,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b512 s[20:35], s[96:99], s0
// GFX11: encoding: [0x30,0x05,0x30,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b512 s[20:35], s[4:7], s101
// GFX11: encoding: [0x02,0x05,0x30,0xf4,0x00,0x00,0x00,0xca]

s_buffer_load_b512 s[20:35], s[4:7], vcc_lo
// GFX11: encoding: [0x02,0x05,0x30,0xf4,0x00,0x00,0x00,0xd4]

s_buffer_load_b512 s[20:35], s[4:7], vcc_hi
// GFX11: encoding: [0x02,0x05,0x30,0xf4,0x00,0x00,0x00,0xd6]

s_buffer_load_b512 s[20:35], s[4:7], m0
// GFX11: encoding: [0x02,0x05,0x30,0xf4,0x00,0x00,0x00,0xfa]

s_buffer_load_b512 s[20:35], s[4:7], 0x0
// GFX11: encoding: [0x02,0x05,0x30,0xf4,0x00,0x00,0x00,0xf8]

s_buffer_load_b512 s[20:35], s[4:7], s0 glc
// GFX11: encoding: [0x02,0x45,0x30,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b512 s[20:35], s[4:7], s0 dlc
// GFX11: encoding: [0x02,0x25,0x30,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b512 s[20:35], s[4:7], s0 glc dlc
// GFX11: encoding: [0x02,0x65,0x30,0xf4,0x00,0x00,0x00,0x00]

s_buffer_load_b512 s[20:35], s[4:7], 0x1234 glc dlc
// GFX11: encoding: [0x02,0x65,0x30,0xf4,0x34,0x12,0x00,0xf8]

s_dcache_inv
// GFX11: encoding: [0x00,0x00,0x84,0xf4,0x00,0x00,0x00,0x00]

s_gl1_inv
// GFX11: encoding: [0x00,0x00,0x80,0xf4,0x00,0x00,0x00,0x00]

s_atc_probe 7, s[4:5], s2
// GFX11: encoding: [0xc2,0x01,0x88,0xf4,0x00,0x00,0x00,0x04]

s_atc_probe 7, s[4:5], 0x64
// GFX11: encoding: [0xc2,0x01,0x88,0xf4,0x64,0x00,0x00,0xf8]

s_atc_probe 7, s[4:5], s9 offset:0x64
// GFX11: encoding: [0xc2,0x01,0x88,0xf4,0x64,0x00,0x00,0x12]

s_atc_probe_buffer 7, s[8:11], s2
// GFX11: encoding: [0xc4,0x01,0x8c,0xf4,0x00,0x00,0x00,0x04]

s_atc_probe_buffer 7, s[8:11], 0x64
// GFX11: encoding: [0xc4,0x01,0x8c,0xf4,0x64,0x00,0x00,0xf8]

s_atc_probe_buffer 7, s[8:11], s9 offset:0x64
// GFX11: encoding: [0xc4,0x01,0x8c,0xf4,0x64,0x00,0x00,0x12]

s_store_dword s1, s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s101, s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword vcc_lo, s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword vcc_hi, s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[6:7], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[100:101], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, vcc, s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[4:5], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[4:5], vcc_lo
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[4:5], vcc_hi
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[4:5], m0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[4:5], 0x0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[4:5], s0 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[4:5], s0 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[4:5], s0 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dword s1, s[4:5], 0x1234 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[4:5], s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[100:101], s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 vcc, s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[6:7], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[100:101], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], vcc, s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], vcc_lo
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], vcc_hi
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], m0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], 0x0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], s0 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], s0 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], s0 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx2 s[2:3], s[4:5], 0x1234 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[8:11], s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[96:99], s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[6:7], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[100:101], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], vcc, s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], vcc_lo
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], vcc_hi
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], m0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], 0x0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], s0 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], s0 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], s0 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_store_dwordx4 s[4:7], s[4:5], 0x1234 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s101, s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword vcc_lo, s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword vcc_hi, s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[12:15], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[96:99], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], vcc_lo
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], vcc_hi
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], m0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], 0x0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], s0 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], s0 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], s0 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dword s1, s[8:11], 0x1234 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[4:5], s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[100:101], s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 vcc, s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[12:15], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[96:99], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], vcc_lo
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], vcc_hi
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], m0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], 0x0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], s0 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], s0 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], s0 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx2 s[2:3], s[8:11], 0x1234 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[8:11], s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[96:99], s[8:11], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[12:15], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[96:99], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], vcc_lo
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], vcc_hi
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], m0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], 0x0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], s0 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], s0 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], s0 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_store_dwordx4 s[4:7], s[8:11], 0x1234 glc dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memrealtime s[10:11]
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memrealtime s[12:13]
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memrealtime s[100:101]
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memrealtime vcc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memtime s[10:11]
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memtime s[12:13]
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memtime s[100:101]
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_memtime vcc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_wb
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_get_waveid_in_workgroup s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_get_waveid_in_workgroup vcc_lo
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_load_dword s5, s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_load_dword s5, s[2:3], s0 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_load_dwordx2 s[100:101], s[2:3], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_load_dwordx2 s[10:11], s[2:3], 0x1 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_load_dwordx4 s[20:23], s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_store_dword s101, s[4:5], s0
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_store_dword s1, s[4:5], 0x123 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_store_dwordx2 s[2:3], s[4:5], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_scratch_store_dwordx4 s[4:7], s[4:5], s0 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_discard s[2:3], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_discard s[2:3], 100
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_discard s[2:3], s7 offset:100
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_discard_x2 s[2:3], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_dcache_discard_x2 s[2:3], 100
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add s5, s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add s5, s[2:3], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add_x2 s[10:11], s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_and s5, s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_and_x2 s[10:11], s[2:3], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap s[10:11], s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap s[10:11], s[2:3], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap_x2 s[20:23], s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap_x2 s[20:23], s[2:3], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_dec_x2 s[10:11], s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_inc_x2 s[10:11], s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_or s5, s[2:3], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smax s5, s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smin s5, s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_sub s5, s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_swap s5, s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umax_x2 s[10:11], s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umin s5, s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_xor s5, s[2:3], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add s5, s[4:7], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add s5, s[4:7], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add_x2 s[10:11], s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_and s101, s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_and_x2 s[10:11], s[8:11], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap s[10:11], s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap s[10:11], s[4:7], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_dec s5, s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_inc s101, s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_inc_x2 s[10:11], s[4:7], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_or s5, s[8:11], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_or_x2 s[10:11], s[96:99], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smax s5, s[4:7], s101
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smax_x2 s[100:101], s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smin s5, s[4:7], 0x64
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smin_x2 s[12:13], s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_sub_x2 s[10:11], s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_swap s5, s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umax s5, s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umin s5, s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_xor s5, s[4:7], s2
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add s5, s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add s5, s[2:3], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add_x2 s[10:11], s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_and s5, s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_and_x2 s[10:11], s[2:3], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap s[10:11], s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap s[10:11], s[2:3], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap_x2 s[20:23], s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap_x2 s[20:23], s[2:3], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_dec_x2 s[10:11], s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_inc_x2 s[10:11], s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_or s5, s[2:3], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smax s5, s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smin s5, s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_sub s5, s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_swap s5, s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umax_x2 s[10:11], s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umin s5, s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_xor s5, s[2:3], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add s5, s[4:7], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add s5, s[4:7], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add_x2 s[10:11], s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_and s101, s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_and_x2 s[10:11], s[8:11], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap s[10:11], s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap s[10:11], s[4:7], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_dec s5, s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_inc s101, s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_inc_x2 s[10:11], s[4:7], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_or s5, s[8:11], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_or_x2 s[10:11], s[96:99], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smax s5, s[4:7], s101 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smax_x2 s[100:101], s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smin s5, s[4:7], 0x64 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smin_x2 s[12:13], s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_sub_x2 s[10:11], s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_swap s5, s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umax s5, s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umin s5, s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_xor s5, s[4:7], s2 glc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add s5, s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add s5, s[2:3], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_add_x2 s[10:11], s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_and s5, s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_and_x2 s[10:11], s[2:3], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap s[10:11], s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap s[10:11], s[2:3], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap_x2 s[20:23], s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_cmpswap_x2 s[20:23], s[2:3], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_dec_x2 s[10:11], s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_inc_x2 s[10:11], s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_or s5, s[2:3], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smax s5, s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_smin s5, s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_sub s5, s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_swap s5, s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umax_x2 s[10:11], s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_umin s5, s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_atomic_xor s5, s[2:3], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add s5, s[4:7], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add s5, s[4:7], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_add_x2 s[10:11], s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_and s101, s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_and_x2 s[10:11], s[8:11], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap s[10:11], s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap s[10:11], s[4:7], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_cmpswap_x2 s[20:23], s[4:7], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_dec s5, s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_inc s101, s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_inc_x2 s[10:11], s[4:7], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_or s5, s[8:11], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_or_x2 s[10:11], s[96:99], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smax s5, s[4:7], s101 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smax_x2 s[100:101], s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smin s5, s[4:7], 0x64 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_smin_x2 s[12:13], s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_sub_x2 s[10:11], s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_swap s5, s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umax s5, s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_umin s5, s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_buffer_atomic_xor s5, s[4:7], s2 dlc
// GFX11-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
