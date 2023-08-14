// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding < %s | FileCheck --check-prefix=GFX1210 %s

v_add_nc_u64 v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1
// GFX1210: v_add_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xfa,0x08,0x08,0x50,0x02,0x53,0x05,0x30]

v_add_nc_u64 v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_add_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x50,0x02,0x50,0x01,0xff]

v_add_nc_u64 v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_add_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0xfa,0x08,0x08,0x50,0x02,0x5f,0x01,0x01]

v_sub_nc_u64 v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1
// GFX1210: v_sub_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xfa,0x08,0x08,0x52,0x02,0x53,0x05,0x30]

v_sub_nc_u64 v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1210: v_sub_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x52,0x02,0x50,0x01,0xff]

v_sub_nc_u64 v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1210: v_sub_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0xfa,0x08,0x08,0x52,0x02,0x5f,0x01,0x01]
