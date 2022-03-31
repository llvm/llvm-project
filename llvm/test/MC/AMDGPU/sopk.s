// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefixes=GCN,SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck --check-prefixes=GCN,SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck --check-prefixes=GCN,VI9,VI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefixes=GCN,VI9,GFX9 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck --check-prefixes=GCN,GFX10 %s

// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck -check-prefix=NOSICIVI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=NOSICIVI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji %s 2>&1 | FileCheck -check-prefix=NOSICIVI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck --check-prefix=NOGFX9 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefix=NOGFX10 --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

s_movk_i32 s2, 0x6
// GCN: s_movk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb0]

s_cmovk_i32 s2, 0x6
// SICI: s_cmovk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb1]
// VI9:  s_cmovk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb0]
// GFX10: s_cmovk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb1]

s_cmpk_eq_i32 s2, 0x6
// SICI: s_cmpk_eq_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb1]
// VI9:  s_cmpk_eq_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb1]
// GFX10: s_cmpk_eq_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb1]

s_cmpk_lg_i32 s2, 0x6
// SICI: s_cmpk_lg_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb2]
// VI9:  s_cmpk_lg_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb1]
// GFX10: s_cmpk_lg_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb2]

s_cmpk_gt_i32 s2, 0x6
// SICI: s_cmpk_gt_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb2]
// VI9:  s_cmpk_gt_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb2]
// GFX10: s_cmpk_gt_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb2]

s_cmpk_ge_i32 s2, 0x6
// SICI: s_cmpk_ge_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb3]
// VI9:  s_cmpk_ge_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb2]
// GFX10: s_cmpk_ge_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb3]

s_cmpk_lt_i32 s2, 0x6
// SICI: s_cmpk_lt_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb3]
// VI9:  s_cmpk_lt_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb3]
// GFX10: s_cmpk_lt_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb3]

s_cmpk_le_i32 s2, 0x6
// SICI: s_cmpk_le_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb4]
// VI9:  s_cmpk_le_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb3]
// GFX10: s_cmpk_le_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb4]

s_cmpk_eq_u32 s2, 0x6
// SICI: s_cmpk_eq_u32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb4]
// VI9:  s_cmpk_eq_u32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb4]
// GFX10: s_cmpk_eq_u32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb4]

s_cmpk_lg_u32 s2, 0x6
// SICI: s_cmpk_lg_u32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb5]
// VI9:  s_cmpk_lg_u32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb4]
// GFX10: s_cmpk_lg_u32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb5]

s_cmpk_gt_u32 s2, 0x6
// SICI: s_cmpk_gt_u32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb5]
// VI9:  s_cmpk_gt_u32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb5]
// GFX10: s_cmpk_gt_u32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb5]

s_cmpk_ge_u32 s2, 0x6
// SICI: s_cmpk_ge_u32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb6]
// VI9:  s_cmpk_ge_u32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb5]
// GFX10: s_cmpk_ge_u32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb6]

s_cmpk_lt_u32 s2, 0x6
// SICI: s_cmpk_lt_u32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb6]
// VI9:  s_cmpk_lt_u32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb6]
// GFX10: s_cmpk_lt_u32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb6]

s_cmpk_le_u32 s2, 0x6
// SICI: s_cmpk_le_u32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb7]
// VI9:  s_cmpk_le_u32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb6]
// GFX10: s_cmpk_le_u32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb7]

s_cmpk_le_u32 s2, 0xFFFF
// SICI: s_cmpk_le_u32 s2, 0xffff ; encoding: [0xff,0xff,0x02,0xb7]
// VI9:  s_cmpk_le_u32 s2, 0xffff ; encoding: [0xff,0xff,0x82,0xb6]
// GFX10: s_cmpk_le_u32 s2, 0xffff ; encoding: [0xff,0xff,0x02,0xb7]

s_addk_i32 s2, 0x6
// SICI: s_addk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb7]
// VI9:  s_addk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb7]
// GFX10: s_addk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb7]

s_mulk_i32 s2, 0x6
// SICI: s_mulk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb8]
// VI9:  s_mulk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x82,0xb7]
// GFX10: s_mulk_i32 s2, 0x6 ; encoding: [0x06,0x00,0x02,0xb8]

s_mulk_i32 s2, -1
// SICI: s_mulk_i32 s2, 0xffff ; encoding: [0xff,0xff,0x02,0xb8]
// VI9:  s_mulk_i32 s2, 0xffff ; encoding: [0xff,0xff,0x82,0xb7]
// GFX10: s_mulk_i32 s2, 0xffff ; encoding: [0xff,0xff,0x02,0xb8]

s_mulk_i32 s2, 0xFFFF
// SICI: s_mulk_i32 s2, 0xffff ; encoding: [0xff,0xff,0x02,0xb8]
// VI9:  s_mulk_i32 s2, 0xffff ; encoding: [0xff,0xff,0x82,0xb7]
// GFX10: s_mulk_i32 s2, 0xffff ; encoding: [0xff,0xff,0x02,0xb8]

s_cbranch_i_fork s[2:3], 0x6
// SICI: s_cbranch_i_fork s[2:3], 6 ; encoding: [0x06,0x00,0x82,0xb8]
// VI9:  s_cbranch_i_fork s[2:3], 6 ; encoding: [0x06,0x00,0x02,0xb8]
// NOGFX10: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// getreg/setreg and hwreg macro
//===----------------------------------------------------------------------===//

// raw number mapped to known HW register
s_getreg_b32 s2, 0x6
// SICI: s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x02,0xb9]

// HW register identifier, non-default offset/width
s_getreg_b32 s2, hwreg(HW_REG_GPR_ALLOC, 1, 31)
// SICI: s_getreg_b32 s2, hwreg(HW_REG_GPR_ALLOC, 1, 31) ; encoding: [0x45,0xf0,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(HW_REG_GPR_ALLOC, 1, 31) ; encoding: [0x45,0xf0,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_GPR_ALLOC, 1, 31) ; encoding: [0x45,0xf0,0x02,0xb9]

// HW register code of unknown HW register, non-default offset/width
s_getreg_b32 s2, hwreg(51, 1, 31)
// SICI: s_getreg_b32 s2, hwreg(51, 1, 31) ; encoding: [0x73,0xf0,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(51, 1, 31) ; encoding: [0x73,0xf0,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(51, 1, 31) ; encoding: [0x73,0xf0,0x02,0xb9]

// HW register code of unknown HW register, default offset/width
s_getreg_b32 s2, hwreg(51)
// SICI: s_getreg_b32 s2, hwreg(51) ; encoding: [0x33,0xf8,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(51) ; encoding: [0x33,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(51) ; encoding: [0x33,0xf8,0x02,0xb9]

// HW register code of unknown HW register, valid symbolic name range but no name available
s_getreg_b32 s2, hwreg(10)
// SICI: s_getreg_b32 s2, hwreg(10) ; encoding: [0x0a,0xf8,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(10) ; encoding: [0x0a,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(10) ; encoding: [0x0a,0xf8,0x02,0xb9]

// HW_REG_SH_MEM_BASES valid starting from GFX9
s_getreg_b32 s2, hwreg(15)
// SICI:  s_getreg_b32 s2, hwreg(15) ; encoding: [0x0f,0xf8,0x02,0xb9]
// VI:    s_getreg_b32 s2, hwreg(15) ; encoding: [0x0f,0xf8,0x82,0xb8]
// GFX9:  s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES) ; encoding: [0x0f,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES) ; encoding: [0x0f,0xf8,0x02,0xb9]

s_getreg_b32 s2, hwreg(16)
// SICI:  s_getreg_b32 s2, hwreg(16) ; encoding: [0x10,0xf8,0x02,0xb9]
// VI:    s_getreg_b32 s2, hwreg(16) ; encoding: [0x10,0xf8,0x82,0xb8]
// GFX9:  s_getreg_b32 s2, hwreg(HW_REG_TBA_LO) ; encoding: [0x10,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_TBA_LO) ; encoding: [0x10,0xf8,0x02,0xb9]

s_getreg_b32 s2, hwreg(17)
// SICI:  s_getreg_b32 s2, hwreg(17) ; encoding: [0x11,0xf8,0x02,0xb9]
// VI:    s_getreg_b32 s2, hwreg(17) ; encoding: [0x11,0xf8,0x82,0xb8]
// GFX9:  s_getreg_b32 s2, hwreg(HW_REG_TBA_HI) ; encoding: [0x11,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_TBA_HI) ; encoding: [0x11,0xf8,0x02,0xb9]

s_getreg_b32 s2, hwreg(18)
// SICI:  s_getreg_b32 s2, hwreg(18) ; encoding: [0x12,0xf8,0x02,0xb9]
// VI:    s_getreg_b32 s2, hwreg(18) ; encoding: [0x12,0xf8,0x82,0xb8]
// GFX9:  s_getreg_b32 s2, hwreg(HW_REG_TMA_LO) ; encoding: [0x12,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_TMA_LO) ; encoding: [0x12,0xf8,0x02,0xb9]

s_getreg_b32 s2, hwreg(19)
// SICI:  s_getreg_b32 s2, hwreg(19) ; encoding: [0x13,0xf8,0x02,0xb9]
// VI:    s_getreg_b32 s2, hwreg(19) ; encoding: [0x13,0xf8,0x82,0xb8]
// GFX9:  s_getreg_b32 s2, hwreg(HW_REG_TMA_HI) ; encoding: [0x13,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_TMA_HI) ; encoding: [0x13,0xf8,0x02,0xb9]

// GFX10+ registers
s_getreg_b32 s2, hwreg(20)
// SICI:  s_getreg_b32 s2, hwreg(20) ; encoding: [0x14,0xf8,0x02,0xb9]
// VI9:   s_getreg_b32 s2, hwreg(20) ; encoding: [0x14,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_LO) ; encoding: [0x14,0xf8,0x02,0xb9]

s_getreg_b32 s2, hwreg(21)
// SICI:  s_getreg_b32 s2, hwreg(21) ; encoding: [0x15,0xf8,0x02,0xb9]
// VI9:   s_getreg_b32 s2, hwreg(21) ; encoding: [0x15,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_HI) ; encoding: [0x15,0xf8,0x02,0xb9]

s_getreg_b32 s2, hwreg(22)
// SICI:  s_getreg_b32 s2, hwreg(22) ; encoding: [0x16,0xf8,0x02,0xb9]
// VI9:   s_getreg_b32 s2, hwreg(22) ; encoding: [0x16,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_XNACK_MASK) ; encoding: [0x16,0xf8,0x02,0xb9]

s_getreg_b32 s2, hwreg(23)
// SICI:  s_getreg_b32 s2, hwreg(23) ; encoding: [0x17,0xf8,0x02,0xb9]
// VI9:   s_getreg_b32 s2, hwreg(23) ; encoding: [0x17,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_HW_ID1) ; encoding: [0x17,0xf8,0x02,0xb9]

s_getreg_b32 s2, hwreg(24)
// SICI:  s_getreg_b32 s2, hwreg(24) ; encoding: [0x18,0xf8,0x02,0xb9]
// VI9:   s_getreg_b32 s2, hwreg(24) ; encoding: [0x18,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_HW_ID2) ; encoding: [0x18,0xf8,0x02,0xb9]

s_getreg_b32 s2, hwreg(25)
// SICI:  s_getreg_b32 s2, hwreg(25) ; encoding: [0x19,0xf8,0x02,0xb9]
// VI9:   s_getreg_b32 s2, hwreg(25) ; encoding: [0x19,0xf8,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_POPS_PACKER) ; encoding: [0x19,0xf8,0x02,0xb9]

// raw number mapped to known HW register
s_setreg_b32 0x6, s2
// SICI: s_setreg_b32 hwreg(HW_REG_LDS_ALLOC, 0, 1), s2 ; encoding: [0x06,0x00,0x82,0xb9]
// VI9:  s_setreg_b32 hwreg(HW_REG_LDS_ALLOC, 0, 1), s2 ; encoding: [0x06,0x00,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_LDS_ALLOC, 0, 1), s2 ; encoding: [0x06,0x00,0x82,0xb9]

// raw number mapped to unknown HW register
s_setreg_b32 0x33, s2
// SICI: s_setreg_b32 hwreg(51, 0, 1), s2 ; encoding: [0x33,0x00,0x82,0xb9]
// VI9:  s_setreg_b32 hwreg(51, 0, 1), s2 ; encoding: [0x33,0x00,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(51, 0, 1), s2 ; encoding: [0x33,0x00,0x82,0xb9]

// raw number mapped to known HW register, default offset/width
s_setreg_b32 0xf803, s2
// SICI: s_setreg_b32 hwreg(HW_REG_TRAPSTS), s2       ; encoding: [0x03,0xf8,0x82,0xb9]
// VI9:  s_setreg_b32 hwreg(HW_REG_TRAPSTS), s2       ; encoding: [0x03,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_TRAPSTS), s2 ; encoding: [0x03,0xf8,0x82,0xb9]

// HW register identifier, default offset/width implied
s_setreg_b32 hwreg(HW_REG_HW_ID), s2
// SICI: s_setreg_b32 hwreg(HW_REG_HW_ID), s2       ; encoding: [0x04,0xf8,0x82,0xb9]
// VI9:  s_setreg_b32 hwreg(HW_REG_HW_ID), s2       ; encoding: [0x04,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_HW_ID1), s2   ; encoding: [0x17,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(HW_REG_HW_ID1), s2
// NOSICIVI: error: specified hardware register is not supported on this GPU
// NOGFX9: error: specified hardware register is not supported on this GPU
// GFX10: s_setreg_b32 hwreg(HW_REG_HW_ID1), s2      ; encoding: [0x17,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(HW_REG_HW_ID2), s2
// NOSICIVI: error: specified hardware register is not supported on this GPU
// NOGFX9: error: specified hardware register is not supported on this GPU
// GFX10: s_setreg_b32 hwreg(HW_REG_HW_ID2), s2      ; encoding: [0x18,0xf8,0x82,0xb9]

// HW register identifier, non-default offset/width
s_setreg_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), s2
// SICI: s_setreg_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), s2       ; encoding: [0x45,0xf0,0x82,0xb9]
// VI9:  s_setreg_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), s2       ; encoding: [0x45,0xf0,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), s2 ; encoding: [0x45,0xf0,0x82,0xb9]

// HW register code of unknown HW register, valid symbolic name range but no name available
s_setreg_b32 hwreg(10), s2
// SICI: s_setreg_b32 hwreg(10), s2      ; encoding: [0x0a,0xf8,0x82,0xb9]
// VI9:  s_setreg_b32 hwreg(10), s2      ; encoding: [0x0a,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(10), s2 ; encoding: [0x0a,0xf8,0x82,0xb9]

// HW_REG_SH_MEM_BASES valid starting from GFX9
s_setreg_b32 hwreg(15), s2
// SICI:  s_setreg_b32 hwreg(15), s2      ; encoding: [0x0f,0xf8,0x82,0xb9]
// VI:    s_setreg_b32 hwreg(15), s2      ; encoding: [0x0f,0xf8,0x02,0xb9]
// GFX9:  s_setreg_b32 hwreg(HW_REG_SH_MEM_BASES), s2 ; encoding: [0x0f,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_SH_MEM_BASES), s2 ; encoding: [0x0f,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(16), s2
// SICI:  s_setreg_b32 hwreg(16), s2      ; encoding: [0x10,0xf8,0x82,0xb9]
// VI:    s_setreg_b32 hwreg(16), s2      ; encoding: [0x10,0xf8,0x02,0xb9]
// GFX9:  s_setreg_b32 hwreg(HW_REG_TBA_LO), s2 ; encoding: [0x10,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_TBA_LO), s2 ; encoding: [0x10,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(17), s2
// SICI:  s_setreg_b32 hwreg(17), s2      ; encoding: [0x11,0xf8,0x82,0xb9]
// VI:    s_setreg_b32 hwreg(17), s2      ; encoding: [0x11,0xf8,0x02,0xb9]
// GFX9:  s_setreg_b32 hwreg(HW_REG_TBA_HI), s2 ; encoding: [0x11,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_TBA_HI), s2 ; encoding: [0x11,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(18), s2
// SICI:  s_setreg_b32 hwreg(18), s2      ; encoding: [0x12,0xf8,0x82,0xb9]
// VI:    s_setreg_b32 hwreg(18), s2      ; encoding: [0x12,0xf8,0x02,0xb9]
// GFX9:  s_setreg_b32 hwreg(HW_REG_TMA_LO), s2 ; encoding: [0x12,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_TMA_LO), s2 ; encoding: [0x12,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(19), s2
// SICI:  s_setreg_b32 hwreg(19), s2      ; encoding: [0x13,0xf8,0x82,0xb9]
// VI:    s_setreg_b32 hwreg(19), s2      ; encoding: [0x13,0xf8,0x02,0xb9]
// GFX9:  s_setreg_b32 hwreg(HW_REG_TMA_HI), s2 ; encoding: [0x13,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_TMA_HI), s2 ; encoding: [0x13,0xf8,0x82,0xb9]

// GFX10+ registers
s_setreg_b32 hwreg(20), s2
// SICI:  s_setreg_b32 hwreg(20), s2      ; encoding: [0x14,0xf8,0x82,0xb9]
// VI9:   s_setreg_b32 hwreg(20), s2      ; encoding: [0x14,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s2 ; encoding: [0x14,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(21), s2
// SICI:  s_setreg_b32 hwreg(21), s2      ; encoding: [0x15,0xf8,0x82,0xb9]
// VI9:   s_setreg_b32 hwreg(21), s2      ; encoding: [0x15,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s2 ; encoding: [0x15,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(22), s2
// SICI:  s_setreg_b32 hwreg(22), s2      ; encoding: [0x16,0xf8,0x82,0xb9]
// VI9:   s_setreg_b32 hwreg(22), s2      ; encoding: [0x16,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_XNACK_MASK), s2 ; encoding: [0x16,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(23), s2
// SICI:  s_setreg_b32 hwreg(23), s2      ; encoding: [0x17,0xf8,0x82,0xb9]
// VI9:   s_setreg_b32 hwreg(23), s2      ; encoding: [0x17,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_HW_ID1), s2      ; encoding: [0x17,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(24), s2
// SICI:  s_setreg_b32 hwreg(24), s2      ; encoding: [0x18,0xf8,0x82,0xb9]
// VI9:   s_setreg_b32 hwreg(24), s2      ; encoding: [0x18,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_HW_ID2), s2      ; encoding: [0x18,0xf8,0x82,0xb9]

s_setreg_b32 hwreg(25), s2
// SICI:  s_setreg_b32 hwreg(25), s2      ; encoding: [0x19,0xf8,0x82,0xb9]
// VI9:   s_setreg_b32 hwreg(25), s2      ; encoding: [0x19,0xf8,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_POPS_PACKER), s2 ; encoding: [0x19,0xf8,0x82,0xb9]

// HW register code, non-default offset/width
s_setreg_b32 hwreg(5, 1, 31), s2
// SICI: s_setreg_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), s2       ; encoding: [0x45,0xf0,0x82,0xb9]
// VI9:  s_setreg_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), s2       ; encoding: [0x45,0xf0,0x02,0xb9]
// GFX10: s_setreg_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), s2 ; encoding: [0x45,0xf0,0x82,0xb9]

// raw number mapped to known HW register
s_setreg_imm32_b32 0x6, 0xff
// SICI: s_setreg_imm32_b32 hwreg(HW_REG_LDS_ALLOC, 0, 1), 0xff ; encoding: [0x06,0x00,0x80,0xba,0xff,0x00,0x00,0x00]
// VI9:  s_setreg_imm32_b32 hwreg(HW_REG_LDS_ALLOC, 0, 1), 0xff ; encoding: [0x06,0x00,0x00,0xba,0xff,0x00,0x00,0x00]
// GFX10: s_setreg_imm32_b32 hwreg(HW_REG_LDS_ALLOC, 0, 1), 0xff ; encoding: [0x06,0x00,0x80,0xba,0xff,0x00,0x00,0x00]

// HW register identifier, non-default offset/width
s_setreg_imm32_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), 0xff
// SICI: s_setreg_imm32_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), 0xff ; encoding: [0x45,0xf0,0x80,0xba,0xff,0x00,0x00,0x00]
// VI9:  s_setreg_imm32_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), 0xff ; encoding: [0x45,0xf0,0x00,0xba,0xff,0x00,0x00,0x00]
// GFX10: s_setreg_imm32_b32 hwreg(HW_REG_GPR_ALLOC, 1, 31), 0xff ; encoding: [0x45,0xf0,0x80,0xba,0xff,0x00,0x00,0x00]

//===----------------------------------------------------------------------===//
// expressions and hwreg macro
//===----------------------------------------------------------------------===//

hwreg=6
s_getreg_b32 s2, hwreg
// SICI: s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x02,0xb9]

x=5
s_getreg_b32 s2, x+1
// SICI: s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x02,0xb9]

x=5
s_getreg_b32 s2, 1+x
// SICI: s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(HW_REG_LDS_ALLOC, 0, 1) ; encoding: [0x06,0x00,0x02,0xb9]

reg=50
offset=2
width=30
s_getreg_b32 s2, hwreg(reg + 1, offset - 1, width + 1)
// SICI: s_getreg_b32 s2, hwreg(51, 1, 31) ; encoding: [0x73,0xf0,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(51, 1, 31) ; encoding: [0x73,0xf0,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(51, 1, 31) ; encoding: [0x73,0xf0,0x02,0xb9]

s_getreg_b32 s2, hwreg(1 + reg, -1 + offset, 1 + width)
// SICI: s_getreg_b32 s2, hwreg(51, 1, 31) ; encoding: [0x73,0xf0,0x02,0xb9]
// VI9:  s_getreg_b32 s2, hwreg(51, 1, 31) ; encoding: [0x73,0xf0,0x82,0xb8]
// GFX10: s_getreg_b32 s2, hwreg(51, 1, 31) ; encoding: [0x73,0xf0,0x02,0xb9]

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

s_endpgm_ordered_ps_done
// GFX9:     s_endpgm_ordered_ps_done ; encoding: [0x00,0x00,0x9e,0xbf]
// NOSICIVI: error: instruction not supported on this GPU
// GFX10: s_endpgm_ordered_ps_done ; encoding: [0x00,0x00,0x9e,0xbf]

s_call_b64 null, 12609
// GFX10: s_call_b64 null, 12609 ; encoding: [0x41,0x31,0x7d,0xbb]
// NOSICIVI: error: instruction not supported on this GPU
// NOGFX9: error: 'null' operand is not supported on this GPU

s_call_b64 s[12:13], 12609
// GFX9:     s_call_b64 s[12:13], 12609 ; encoding: [0x41,0x31,0x8c,0xba]
// NOSICIVI: error: instruction not supported on this GPU
// GFX10: s_call_b64 s[12:13], 12609 ; encoding: [0x41,0x31,0x0c,0xbb]

s_call_b64 s[100:101], 12609
// GFX9:     s_call_b64 s[100:101], 12609 ; encoding: [0x41,0x31,0xe4,0xba]
// NOSICIVI: error: instruction not supported on this GPU
// GFX10: s_call_b64 s[100:101], 12609 ; encoding: [0x41,0x31,0x64,0xbb]

s_call_b64 s[10:11], 49617
// GFX9:     s_call_b64 s[10:11], 49617 ; encoding: [0xd1,0xc1,0x8a,0xba]
// NOSICIVI: error: instruction not supported on this GPU
// GFX10: s_call_b64 s[10:11], 49617 ; encoding: [0xd1,0xc1,0x0a,0xbb]

offset = 4
s_call_b64 s[0:1], offset + 4
// GFX9:     s_call_b64 s[0:1], 8            ; encoding: [0x08,0x00,0x80,0xba]
// NOSICIVI: error: instruction not supported on this GPU
// GFX10: s_call_b64 s[0:1], 8 ; encoding: [0x08,0x00,0x00,0xbb]

offset = 4
s_call_b64 s[0:1], 4 + offset
// GFX9:     s_call_b64 s[0:1], 8            ; encoding: [0x08,0x00,0x80,0xba]
// NOSICIVI: error: instruction not supported on this GPU
// GFX10: s_call_b64 s[0:1], 8 ; encoding: [0x08,0x00,0x00,0xbb]
