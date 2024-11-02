// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck --check-prefix=VI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck -check-prefix=GFX11 %s

// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck -check-prefixes=GCN,SICIVI-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefixes=GCN,SICIVI-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefixes=GCN,SICIVI-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefixes=GCN,GFX9-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck -check-prefixes=GCN,GFX10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck -check-prefixes=GCN,GFX11-ERR --implicit-check-not=error: %s

s_setreg_b32  0x1f803, s2
// GCN: error: invalid immediate: only 16-bit values are legal

s_setreg_b32  typo(0x40), s2
// GCN: error: expected a hwreg macro or an absolute expression

s_setreg_b32  hwreg(0x40), s2
// GCN: error: invalid code of hardware register: only 6-bit values are legal

s_setreg_b32  hwreg(HW_REG_WRONG), s2
// GCN: error: expected a register name or an absolute expression

s_setreg_b32  hwreg(1 2,3), s2
// GCN: error: expected a comma or a closing parenthesis

s_setreg_b32  hwreg(1,2 3), s2
// GCN: error: expected a comma

s_setreg_b32  hwreg(1,2,3, s2
// GCN: error: expected a closing parenthesis

s_setreg_b32  hwreg(3,32,32), s2
// GCN: error: invalid bit offset: only 5-bit values are legal

s_setreg_b32  hwreg(3,0,33), s2
// GCN: error: invalid bitfield width: only values from 1 to 32 are legal

s_setreg_imm32_b32  0x1f803, 0xff
// GCN: error: invalid immediate: only 16-bit values are legal

s_setreg_imm32_b32  hwreg(3,0,33), 0xff
// GCN: error: invalid bitfield width: only values from 1 to 32 are legal

s_getreg_b32  s2, hwreg(3,32,32)
// GCN: error: invalid bit offset: only 5-bit values are legal

s_cbranch_i_fork s[2:3], 0x6
// SICI: s_cbranch_i_fork s[2:3], 6 ; encoding: [0x06,0x00,0x82,0xb8]
// GFX10-ERR: error: instruction not supported on this GPU
// GFX9: s_cbranch_i_fork s[2:3], 6 ; encoding: [0x06,0x00,0x02,0xb8]
// VI: s_cbranch_i_fork s[2:3], 6 ; encoding: [0x06,0x00,0x02,0xb8]
// GFX11-ERR: error: instruction not supported on this GPU

s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES)
// GFX10:  s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES) ; encoding: [0x0f,0xf8,0x02,0xb9]
// SICIVI-ERR: error: specified hardware register is not supported on this GPU
// GFX9: s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES) ; encoding: [0x0f,0xf8,0x82,0xb8]
// GFX11: s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES) ; encoding: [0x0f,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_TBA_LO)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_TBA_LO) ; encoding: [0x10,0xf8,0x02,0xb9]
// SICIVI-ERR: error: specified hardware register is not supported on this GPU
// GFX11-ERR: error: specified hardware register is not supported on this GPU
// GFX9:     s_getreg_b32 s2, hwreg(HW_REG_TBA_LO)   ; encoding: [0x10,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_TBA_HI)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_TBA_HI) ; encoding: [0x11,0xf8,0x02,0xb9]
// SICIVI-ERR: error: specified hardware register is not supported on this GPU
// GFX11-ERR: error: specified hardware register is not supported on this GPU
// GFX9:     s_getreg_b32 s2, hwreg(HW_REG_TBA_HI)   ; encoding: [0x11,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_TMA_LO)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_TMA_LO) ; encoding: [0x12,0xf8,0x02,0xb9]
// SICIVI-ERR: error: specified hardware register is not supported on this GPU
// GFX11-ERR: error: specified hardware register is not supported on this GPU
// GFX9:     s_getreg_b32 s2, hwreg(HW_REG_TMA_LO)   ; encoding: [0x12,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_TMA_HI)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_TMA_HI) ; encoding: [0x13,0xf8,0x02,0xb9]
// SICIVI-ERR: error: specified hardware register is not supported on this GPU
// GFX11-ERR: error: specified hardware register is not supported on this GPU
// GFX9:     s_getreg_b32 s2, hwreg(HW_REG_TMA_HI)   ; encoding: [0x13,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_LO)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_LO) ; encoding: [0x14,0xf8,0x02,0xb9]
// SICIVI-ERR: error: specified hardware register is not supported on this GPU
// GFX9-ERR: error: specified hardware register is not supported on this GPU
// GFX11: s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_LO) ; encoding: [0x14,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_HI)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_HI) ; encoding: [0x15,0xf8,0x02,0xb9]
// SICIVI-ERR: error: specified hardware register is not supported on this GPU
// GFX9-ERR: error: specified hardware register is not supported on this GPU
// GFX11: s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_HI) ; encoding: [0x15,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_XNACK_MASK)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_XNACK_MASK) ; encoding: [0x16,0xf8,0x02,0xb9]
// SICIVI-ERR: error: specified hardware register is not supported on this GPU
// GFX9-ERR: error: specified hardware register is not supported on this GPU
// GFX11-ERR: error: specified hardware register is not supported on this GPU

s_getreg_b32 s2, hwreg(HW_REG_POPS_PACKER)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_POPS_PACKER) ; encoding: [0x19,0xf8,0x02,0xb9]
// SICIVI-ERR: error: specified hardware register is not supported on this GPU
// GFX9-ERR: error: specified hardware register is not supported on this GPU
// GFX11-ERR: error: specified hardware register is not supported on this GPU

s_cmpk_le_u32 s2, -1
// GCN: error: invalid operand for instruction

s_cmpk_le_u32 s2, 0x1ffff
// GCN: error: invalid operand for instruction

s_cmpk_le_u32 s2, 0x10000
// GCN: error: invalid operand for instruction

s_mulk_i32 s2, 0xFFFFFFFFFFFF0000
// GCN: error: invalid operand for instruction

s_mulk_i32 s2, 0x10000
// GCN: error: invalid operand for instruction
