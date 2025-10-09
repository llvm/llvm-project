// RUN: llvm-mc -triple=amdgcn -show-encoding -mcpu=gfx1200 %s | FileCheck --check-prefix=GFX12 %s

s_addk_i32 s0, 0x1234
// GFX12: s_addk_co_i32 s0, 0x1234                                          ; encoding: [0x34,0x12,0x80,0xb7]

s_getreg_b32 s0, hwreg(HW_REG_MODE)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_MODE)                          ; encoding: [0x01,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_STATUS)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_STATUS)                        ; encoding: [0x02,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_STATE_PRIV)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_STATE_PRIV)                    ; encoding: [0x04,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_GPR_ALLOC)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_GPR_ALLOC)                     ; encoding: [0x05,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_LDS_ALLOC)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_LDS_ALLOC)                     ; encoding: [0x06,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_EXCP_FLAG_PRIV)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_EXCP_FLAG_PRIV)                ; encoding: [0x11,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_EXCP_FLAG_USER)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_EXCP_FLAG_USER)                ; encoding: [0x12,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_TRAP_CTRL)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_TRAP_CTRL)                     ; encoding: [0x13,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_SCRATCH_BASE_LO)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_SCRATCH_BASE_LO)               ; encoding: [0x14,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_SCRATCH_BASE_HI)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_SCRATCH_BASE_HI)               ; encoding: [0x15,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_HW_ID1)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_HW_ID1)                        ; encoding: [0x17,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_HW_ID2)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_HW_ID2)                        ; encoding: [0x18,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_DVGPR_ALLOC_LO)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_DVGPR_ALLOC_LO)                ; encoding: [0x1f,0xf8,0x80,0xb8]

s_getreg_b32 s0, hwreg(HW_REG_DVGPR_ALLOC_HI)
// GFX12: s_getreg_b32 s0, hwreg(HW_REG_WAVE_DVGPR_ALLOC_HI)                ; encoding: [0x20,0xf8,0x80,0xb8]
