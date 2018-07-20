// RUN: not llvm-mc -arch=amdgcn %s 2>&1 | FileCheck -check-prefix=GCN %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=SI-ERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=VI-ERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s 2>&1 | FileCheck -check-prefix=GFX9 %s

s_setreg_b32  0x1f803, s2
// GCN: error: invalid immediate: only 16-bit values are legal

s_setreg_b32  hwreg(0x40), s2
// GCN: error: invalid code of hardware register: only 6-bit values are legal

s_setreg_b32  hwreg(HW_REG_WRONG), s2
// GCN: error: invalid symbolic name of hardware register

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

s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES)
// SI-ERR: error: invalid symbolic name of hardware register
// VI-ERR: error: invalid symbolic name of hardware register
// GFX9: s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES) ; encoding: [0x0f,0xf8,0x82,0xb8]

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
