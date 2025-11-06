// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s

buffer_load_dword v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_b32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x02,0x03]

buffer_load_dwordx2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_load_b64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x05,0x02,0x03]

buffer_load_dwordx3 v[5:7], off, s[8:11], s3 offset:4095
// GFX11: buffer_load_b96 v[5:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x05,0x02,0x03]

buffer_load_dwordx4 v[5:8], off, s[8:11], s3 offset:4095
// GFX11: buffer_load_b128 v[5:8], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x05,0x02,0x03]

buffer_load_short_d16 v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_b16 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x02,0x03]

buffer_load_format_d16_x v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_format_x v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x02,0x03]

buffer_load_format_d16_xy v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x02,0x03]

buffer_load_format_d16_xyz v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x05,0x02,0x03]

buffer_load_format_d16_xyzw v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x05,0x02,0x03]

buffer_load_short_d16_hi v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_hi_b16 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x02,0x03]

buffer_load_format_d16_hi_x v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe0,0x00,0x05,0x02,0x03]

buffer_load_sbyte_d16_hi v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_hi_i8 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x02,0x03]

buffer_load_ubyte_d16_hi v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_hi_u8 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x02,0x03]

buffer_load_sbyte_d16 v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_i8 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x05,0x02,0x03]

buffer_load_ubyte_d16 v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_d16_u8 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x05,0x02,0x03]

buffer_load_sbyte v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_i8 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x02,0x03]

buffer_load_sshort v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_i16 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x02,0x03]

buffer_load_ubyte v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_u8 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x02,0x03]

buffer_load_ushort v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_load_u16 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x02,0x03]

buffer_store_byte v1, off, s[12:15], s4 offset:4095
// GFX11: buffer_store_b8 v1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x03,0x04]

buffer_store_short v1, off, s[12:15], s4 offset:4095
// GFX11: buffer_store_b16 v1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x03,0x04]

buffer_store_dword v1, off, s[12:15], s4 offset:4095
// GFX11: buffer_store_b32 v1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x03,0x04]

buffer_store_dwordx2 v[1:2], off, s[12:15], s4 offset:4095
// GFX11: buffer_store_b64 v[1:2], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x03,0x04]

buffer_store_dwordx3 v[1:3], off, s[12:15], s4 offset:4095
// GFX11: buffer_store_b96 v[1:3], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x03,0x04]

buffer_store_dwordx4 v[1:4], off, s[12:15], s4 offset:4095
// GFX11: buffer_store_b128 v[1:4], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x01,0x03,0x04]

buffer_store_format_d16_x v1, off, s[12:15], s4 offset:4095
// GFX11: buffer_store_d16_format_x v1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x03,0x04]

buffer_store_format_d16_xy v1, off, s[12:15], s4 offset:4095
// GFX11: buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x03,0x04]

buffer_store_format_d16_xyz v[1:2], off, s[12:15], s4 offset:4095
// GFX11: buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x01,0x03,0x04]

buffer_store_format_d16_xyzw v[1:2], off, s[12:15], s4 offset:4095
// GFX11: buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x01,0x03,0x04]

buffer_store_byte_d16_hi v1, off, s[12:15], s4 offset:4095
// GFX11: buffer_store_d16_hi_b8 v1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x01,0x03,0x04]

buffer_store_short_d16_hi v1, off, s[12:15], s4 offset:4095
// GFX11: buffer_store_d16_hi_b16 v1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x01,0x03,0x04]

buffer_store_format_d16_hi_x v1, off, s[12:15], s4 offset:4095
// GFX11: buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe0,0x00,0x01,0x03,0x04]

buffer_atomic_add v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_add_u32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xd4,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_add_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_add_u64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_and v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_and_b32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xf0,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_and_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_and_b64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_cmpswap v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_cmpswap_b32 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xd0,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_cmpswap_x2 v[5:8], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_cmpswap_b64 v[5:8], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_fcmpswap v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_cmpswap_f32 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_csub v255, off, s[8:11], s3 offset:4095 glc
// GFX11: buffer_atomic_csub_u32 v255, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0xdc,0xe0,0x00,0xff,0x02,0x03]

buffer_atomic_dec v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_dec_u32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_dec_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_dec_u64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_inc v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_inc_u32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xfc,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_inc_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_inc_u64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_fmax v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_max_f32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_smax v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_max_i32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xe8,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_smax_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_max_i64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_umax v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_max_u32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xec,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_umax_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_max_u64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_fmin v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_min_f32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_smin v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_min_i32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xe0,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_smin_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_min_i64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_umin v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_min_u32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xe4,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_umin_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_min_u64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_or v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_or_b32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xf4,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_or_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_or_b64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_sub v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_sub_u32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xd8,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_sub_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_sub_u64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_swap v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_swap_b32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xcc,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_swap_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_swap_b64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe1,0x00,0x05,0x02,0x03]

buffer_atomic_xor v5, off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_xor_b32 v5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xf8,0xe0,0x00,0x05,0x02,0x03]

buffer_atomic_xor_x2 v[5:6], off, s[8:11], s3 offset:4095
// GFX11: buffer_atomic_xor_b64 v[5:6], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe1,0x00,0x05,0x02,0x03]
