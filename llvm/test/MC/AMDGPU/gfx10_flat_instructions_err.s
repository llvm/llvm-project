// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=GFX1010,GFX10 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1030 %s 2>&1 | FileCheck --check-prefixes=GFX1030,GFX10 --implicit-check-not=error: %s

global_atomic_add v2, v4, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_atomic_add v0, v2, v4, null glc
// GFX10: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_add_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_atomic_add_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:42: error: invalid operand for instruction

global_atomic_and v2, v4, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_atomic_and v0, v2, v4, null
// GFX10: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_and_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_atomic_and_x2 v0, v2, v[4:5], null
// GFX10: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

global_atomic_cmpswap v2, v[4:5], null
// GFX10: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_cmpswap v0, v2, v[4:5], null
// GFX10: :[[@LINE-1]]:39: error: invalid operand for instruction

global_atomic_cmpswap_x2 v2, v[4:7], null
// GFX10: :[[@LINE-1]]:38: error: invalid operand for instruction

global_atomic_cmpswap_x2 v[0:1], v2, v[4:7], null
// GFX10: :[[@LINE-1]]:46: error: invalid operand for instruction

global_atomic_csub v2, v4, null
// GFX1010: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX1030: :[[@LINE-2]]:28: error: invalid operand for instruction

global_atomic_csub v0, v2, v4, null
// GFX1010: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX1030: :[[@LINE-2]]:32: error: invalid operand for instruction

global_atomic_dec v2, v4, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_atomic_dec v0, v2, v4, null
// GFX10: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_dec_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_atomic_dec_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:42: error: invalid operand for instruction

global_atomic_fcmpswap v2, v[4:5], null
// GFX10: :[[@LINE-1]]:36: error: invalid operand for instruction

global_atomic_fcmpswap v0, v2, v[4:5], null
// GFX10: :[[@LINE-1]]:40: error: invalid operand for instruction

global_atomic_fcmpswap_x2 v2, v[4:7], null
// GFX10: :[[@LINE-1]]:39: error: invalid operand for instruction

global_atomic_fcmpswap_x2 v[0:1], v2, v[4:7], null
// GFX10: :[[@LINE-1]]:47: error: invalid operand for instruction

global_atomic_fmax v2, v4, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_atomic_fmax v0, v2, v4, null
// GFX10: :[[@LINE-1]]:32: error: invalid operand for instruction

global_atomic_fmax_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_fmax_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_fmin v2, v4, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_atomic_fmin v0, v2, v4, null
// GFX10: :[[@LINE-1]]:32: error: invalid operand for instruction

global_atomic_fmin_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_fmin_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_inc v2, v4, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_atomic_inc v0, v2, v4, null
// GFX10: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_inc_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_atomic_inc_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:42: error: invalid operand for instruction

global_atomic_or v2, v4, null
// GFX10: :[[@LINE-1]]:26: error: invalid operand for instruction

global_atomic_or v0, v2, v4, null
// GFX10: :[[@LINE-1]]:30: error: invalid operand for instruction

global_atomic_or_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:33: error: invalid operand for instruction

global_atomic_or_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:41: error: invalid operand for instruction

global_atomic_smax v2, v4, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_atomic_smax v0, v2, v4, null
// GFX10: :[[@LINE-1]]:32: error: invalid operand for instruction

global_atomic_smax_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_smax_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_smin v2, v4, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_atomic_smin v0, v2, v4, null
// GFX10: :[[@LINE-1]]:32: error: invalid operand for instruction

global_atomic_smin_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_smin_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_sub v2, v4, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_atomic_sub v0, v2, v4, null
// GFX10: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_sub_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_atomic_sub_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:42: error: invalid operand for instruction

global_atomic_swap v2, v4, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_atomic_swap v0, v2, v4, null
// GFX10: :[[@LINE-1]]:32: error: invalid operand for instruction

global_atomic_swap_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_swap_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_umax v2, v4, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_atomic_umax v0, v2, v4, null
// GFX10: :[[@LINE-1]]:32: error: invalid operand for instruction

global_atomic_umax_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_umax_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_umin v2, v4, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_atomic_umin v0, v2, v4, null
// GFX10: :[[@LINE-1]]:32: error: invalid operand for instruction

global_atomic_umin_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:35: error: invalid operand for instruction

global_atomic_umin_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:43: error: invalid operand for instruction

global_atomic_xor v2, v4, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_atomic_xor v0, v2, v4, null
// GFX10: :[[@LINE-1]]:31: error: invalid operand for instruction

global_atomic_xor_x2 v2, v[4:5], null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_atomic_xor_x2 v[0:1], v2, v[4:5], null
// GFX10: :[[@LINE-1]]:42: error: invalid operand for instruction

global_load_dword v0, v4, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_load_dwordx2 v[0:1], v4, null
// GFX10: :[[@LINE-1]]:33: error: invalid operand for instruction

global_load_dwordx3 v[0:2], v4, null
// GFX10: :[[@LINE-1]]:33: error: invalid operand for instruction

global_load_dwordx4 v[0:3], v4, null
// GFX10: :[[@LINE-1]]:33: error: invalid operand for instruction

global_load_sbyte v0, v2, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_load_sbyte_d16 v0, v2, null
// GFX10: :[[@LINE-1]]:31: error: invalid operand for instruction

global_load_sbyte_d16_hi v0, v2, null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_load_short_d16 v0, v2, null
// GFX10: :[[@LINE-1]]:31: error: invalid operand for instruction

global_load_short_d16_hi v0, v2, null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_load_sshort v0, v2, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_load_ubyte v0, v2, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_load_ubyte_d16 v0, v2, null
// GFX10: :[[@LINE-1]]:31: error: invalid operand for instruction

global_load_ubyte_d16_hi v0, v2, null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_load_ushort v0, v2, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_store_byte v0, v2, null
// GFX10: :[[@LINE-1]]:27: error: invalid operand for instruction

global_store_byte_d16_hi v0, v2, null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_store_dword v0, v2, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_store_dwordx2 v0, v[2:3], null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_store_dwordx3 v0, v[2:4], null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_store_dwordx4 v0, v[2:5], null
// GFX10: :[[@LINE-1]]:34: error: invalid operand for instruction

global_store_short v0, v2, null
// GFX10: :[[@LINE-1]]:28: error: invalid operand for instruction

global_store_short_d16_hi v0, v2, null
// GFX10: :[[@LINE-1]]:35: error: invalid operand for instruction
