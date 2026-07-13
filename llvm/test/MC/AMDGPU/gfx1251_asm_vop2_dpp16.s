// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1251 -show-encoding < %s | FileCheck --check-prefix=GFX1251 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_add_nc_u64 v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1
// GFX1251: v_add_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xfa,0x08,0x08,0x50,0x02,0x53,0x05,0x30]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_add_nc_u64 v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1
// GFX1250-ERR-NEXT:{{^}}                                    ^

v_add_nc_u64 v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_add_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x50,0x02,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_add_nc_u64 v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1250-ERR-NEXT:{{^}}                                    ^

v_add_nc_u64 v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1251: v_add_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0xfa,0x08,0x08,0x50,0x02,0x5f,0x01,0x01]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_add_nc_u64 v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1250-ERR-NEXT:{{^}}                                    ^

v_sub_nc_u64 v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1
// GFX1251: v_sub_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1 ; encoding: [0xfa,0x08,0x08,0x52,0x02,0x53,0x05,0x30]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_sub_nc_u64 v[4:5], v[2:3], v[4:5] row_share:3 row_mask:0x3 bank_mask:0x0 fi:1
// GFX1250-ERR-NEXT:{{^}}                                    ^

v_sub_nc_u64 v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1251: v_sub_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x52,0x02,0x50,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_sub_nc_u64 v[4:5], v[2:3], v[4:5] row_share:0 row_mask:0xf bank_mask:0xf
// GFX1250-ERR-NEXT:{{^}}                                    ^

v_sub_nc_u64 v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1251: v_sub_nc_u64_dpp v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1 ; encoding: [0xfa,0x08,0x08,0x52,0x02,0x5f,0x01,0x01]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_sub_nc_u64 v[4:5], v[2:3], v[4:5] row_share:15 row_mask:0x0 bank_mask:0x1
// GFX1250-ERR-NEXT:{{^}}                                    ^

v_fmac_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1251: v_fmac_f64_dpp v[4:5], v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x2e,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_fmac_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                  ^

v_add_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1251: v_add_f64_dpp v[4:5], v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x04,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_add_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                 ^

v_mul_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1251: v_mul_f64_dpp v[4:5], v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x0c,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_mul_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                 ^

v_max_num_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1251: v_max_num_f64_dpp v[4:5], v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x1c,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_max_num_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                     ^

v_min_num_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1251: v_min_num_f64_dpp v[4:5], v[2:3], v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x1a,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_min_num_f64 v[4:5], v[2:3], v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                     ^

v_lshlrev_b64 v[4:5], v2, v[4:5] row_share:1
// GFX1251: v_lshlrev_b64_dpp v[4:5], v2, v[4:5] row_share:1 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x08,0x08,0x3e,0x02,0x51,0x01,0xff]
// GFX1250-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_lshlrev_b64 v[4:5], v2, v[4:5] row_share:1
// GFX1250-ERR-NEXT:{{^}}                                 ^
