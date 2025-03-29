// RUN: not llvm-mc -triple=amdgcn -show-encoding %s | FileCheck --check-prefix=SICI %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck --check-prefix=SICI %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga -show-encoding %s | FileCheck --check-prefix=VI %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck -check-prefix=GFX11 %s

// RUN: not llvm-mc -triple=amdgcn %s 2>&1 | FileCheck -check-prefixes=GCN,SICIVI-ERR --implicit-check-not=error: --strict-whitespace %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefixes=GCN,SICIVI-ERR --implicit-check-not=error: --strict-whitespace %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefixes=GCN,SICIVI-ERR --implicit-check-not=error: --strict-whitespace %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefixes=GCN,GFX9-ERR --implicit-check-not=error: --strict-whitespace %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck -check-prefixes=GCN,GFX10-ERR --implicit-check-not=error: --strict-whitespace %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck -check-prefixes=GCN,GFX11-ERR --implicit-check-not=error: --strict-whitespace %s

s_setreg_b32  0x1f803, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid immediate: only 16-bit values are legal
// GCN-NEXT: {{^}}s_setreg_b32  0x1f803, s2
// GCN-NEXT: {{^}}              ^

s_setreg_b32  typo(0x40), s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: expected a hwreg macro, structured immediate or an absolute expression
// GCN-NEXT: {{^}}s_setreg_b32  typo(0x40), s2
// GCN-NEXT: {{^}}              ^

s_setreg_b32  hwreg(0x40), s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid hardware register: only 6-bit values are legal
// GCN-NEXT: {{^}}s_setreg_b32  hwreg(0x40), s2
// GCN-NEXT: {{^}}                    ^

s_setreg_b32  {id: 0x40}, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid hardware register: only 6-bit values are legal
// GCN-NEXT: {{^}}s_setreg_b32  {id: 0x40}, s2
// GCN-NEXT: {{^}}                   ^

s_setreg_b32  hwreg(HW_REG_WRONG), s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: expected a register name or an absolute expression
// GCN-NEXT: {{^}}s_setreg_b32  hwreg(HW_REG_WRONG), s2
// GCN-NEXT: {{^}}                    ^

s_setreg_b32  hwreg(1 2,3), s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: expected a comma or a closing parenthesis
// GCN-NEXT: {{^}}s_setreg_b32  hwreg(1 2,3), s2
// GCN-NEXT: {{^}}                      ^

s_setreg_b32  hwreg(1,2 3), s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: expected a comma
// GCN-NEXT: {{^}}s_setreg_b32  hwreg(1,2 3), s2
// GCN-NEXT: {{^}}                        ^

s_setreg_b32  hwreg(1,2,3, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: expected a closing parenthesis
// GCN-NEXT: {{^}}s_setreg_b32  hwreg(1,2,3, s2
// GCN-NEXT: {{^}}                         ^

s_setreg_b32  {id: 1 offset: 2, size: 3}, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: comma or closing brace expected
// GCN-NEXT: {{^}}s_setreg_b32  {id: 1 offset: 2, size: 3}, s2
// GCN-NEXT: {{^}}                     ^

s_setreg_b32  {id: 1 offset: 2, size: 3}, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: comma or closing brace expected
// GCN-NEXT: {{^}}s_setreg_b32  {id: 1 offset: 2, size: 3}, s2
// GCN-NEXT: {{^}}                     ^

s_setreg_b32  {id 1, offset: 2, size: 3}, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: colon expected
// GCN-NEXT: {{^}}s_setreg_b32  {id 1, offset: 2, size: 3}, s2
// GCN-NEXT: {{^}}                  ^

s_setreg_b32  {id: 1, offset: 2, size: 3, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: colon expected
// GCN-NEXT: {{^}}s_setreg_b32  {id: 1, offset: 2, size: 3, s2
// GCN-NEXT: {{^}}                                            ^

s_setreg_b32  {id: 1, offset: 2, blah: 3}, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: unknown field
// GCN-NEXT: {{^}}s_setreg_b32  {id: 1, offset: 2, blah: 3}, s2
// GCN-NEXT: {{^}}                                 ^

s_setreg_b32  {id: 1, id: 2}, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: duplicate field
// GCN-NEXT: {{^}}s_setreg_b32  {id: 1, id: 2}, s2
// GCN-NEXT: {{^}}                      ^

s_setreg_b32  hwreg(3,32,32), s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid bit offset: only 5-bit values are legal
// GCN-NEXT: {{^}}s_setreg_b32  hwreg(3,32,32), s2
// GCN-NEXT: {{^}}                      ^

s_setreg_b32  {id: 3, offset: 32, size: 32}, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid bit offset: only 5-bit values are legal
// GCN-NEXT: {{^}}s_setreg_b32  {id: 3, offset: 32, size: 32}, s2
// GCN-NEXT: {{^}}                              ^

s_setreg_b32  hwreg(3,0,33), s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid bitfield width: only values from 1 to 32 are legal
// GCN-NEXT: {{^}}s_setreg_b32  hwreg(3,0,33), s2
// GCN-NEXT: {{^}}                        ^

s_setreg_b32  {id: 3, offset: 0, size: 33}, s2
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid bitfield width: only values from 1 to 32 are legal
// GCN-NEXT: {{^}}s_setreg_b32  {id: 3, offset: 0, size: 33}, s2
// GCN-NEXT: {{^}}                                       ^

s_setreg_imm32_b32  0x1f803, 0xff
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid immediate: only 16-bit values are legal
// GCN-NEXT: {{^}}s_setreg_imm32_b32  0x1f803, 0xff
// GCN-NEXT: {{^}}                    ^

s_setreg_imm32_b32  hwreg(3,0,33), 0xff
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid bitfield width: only values from 1 to 32 are legal
// GCN-NEXT: {{^}}s_setreg_imm32_b32  hwreg(3,0,33), 0xff
// GCN-NEXT: {{^}}                              ^

s_setreg_imm32_b32  {id: 3, offset: 0, size: 33}, 0xff
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid bitfield width: only values from 1 to 32 are legal
// GCN-NEXT: {{^}}s_setreg_imm32_b32  {id: 3, offset: 0, size: 33}, 0xff
// GCN-NEXT: {{^}}                                             ^

s_getreg_b32  s2, hwreg(3,32,32)
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid bit offset: only 5-bit values are legal
// GCN-NEXT: {{^}}s_getreg_b32  s2, hwreg(3,32,32)
// GCN-NEXT: {{^}}                          ^

s_getreg_b32  s2, {id: 3, offset: 32, size: 32}
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid bit offset: only 5-bit values are legal
// GCN-NEXT: {{^}}s_getreg_b32  s2, {id: 3, offset: 32, size: 32}
// GCN-NEXT: {{^}}                                  ^

s_cbranch_i_fork s[2:3], 0x6
// SICI: s_cbranch_i_fork s[2:3], 6 ; encoding: [0x06,0x00,0x82,0xb8]
// GFX10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: s_cbranch_i_fork s[2:3], 6 ; encoding: [0x06,0x00,0x02,0xb8]
// VI: s_cbranch_i_fork s[2:3], 6 ; encoding: [0x06,0x00,0x02,0xb8]
// GFX11-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES)
// GFX10:  s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES) ; encoding: [0x0f,0xf8,0x02,0xb9]
// SICIVI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// SICIVI-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES)
// SICIVI-ERR-NEXT: {{^}}                       ^
// GFX9: s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES) ; encoding: [0x0f,0xf8,0x82,0xb8]
// GFX11: s_getreg_b32 s2, hwreg(HW_REG_SH_MEM_BASES) ; encoding: [0x0f,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_TBA_LO)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_TBA_LO) ; encoding: [0x10,0xf8,0x02,0xb9]
// SICIVI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// SICIVI-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_TBA_LO)
// SICIVI-ERR-NEXT: {{^}}                       ^
// GFX11-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX11-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_TBA_LO)
// GFX11-ERR-NEXT: {{^}}                       ^
// GFX9:     s_getreg_b32 s2, hwreg(HW_REG_TBA_LO)   ; encoding: [0x10,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_TBA_HI)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_TBA_HI) ; encoding: [0x11,0xf8,0x02,0xb9]
// SICIVI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// SICIVI-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_TBA_HI)
// SICIVI-ERR-NEXT: {{^}}                       ^
// GFX11-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX11-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_TBA_HI)
// GFX11-ERR-NEXT: {{^}}                       ^
// GFX9:     s_getreg_b32 s2, hwreg(HW_REG_TBA_HI)   ; encoding: [0x11,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_TMA_LO)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_TMA_LO) ; encoding: [0x12,0xf8,0x02,0xb9]
// SICIVI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// SICIVI-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_TMA_LO)
// SICIVI-ERR-NEXT: {{^}}                       ^
// GFX11-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX11-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_TMA_LO)
// GFX11-ERR-NEXT: {{^}}                       ^
// GFX9:     s_getreg_b32 s2, hwreg(HW_REG_TMA_LO)   ; encoding: [0x12,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_TMA_HI)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_TMA_HI) ; encoding: [0x13,0xf8,0x02,0xb9]
// SICIVI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// SICIVI-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_TMA_HI)
// SICIVI-ERR-NEXT: {{^}}                       ^
// GFX11-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX11-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_TMA_HI)
// GFX11-ERR-NEXT: {{^}}                       ^
// GFX9:     s_getreg_b32 s2, hwreg(HW_REG_TMA_HI)   ; encoding: [0x13,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_LO)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_LO) ; encoding: [0x14,0xf8,0x02,0xb9]
// SICIVI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// SICIVI-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_LO)
// SICIVI-ERR-NEXT: {{^}}                       ^
// GFX9-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX9-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_LO)
// GFX9-ERR-NEXT: {{^}}                       ^
// GFX11: s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_LO) ; encoding: [0x14,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_HI)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_HI) ; encoding: [0x15,0xf8,0x02,0xb9]
// SICIVI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// SICIVI-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_HI)
// SICIVI-ERR-NEXT: {{^}}                       ^
// GFX9-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX9-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_HI)
// GFX9-ERR-NEXT: {{^}}                       ^
// GFX11: s_getreg_b32 s2, hwreg(HW_REG_FLAT_SCR_HI) ; encoding: [0x15,0xf8,0x82,0xb8]

s_getreg_b32 s2, hwreg(HW_REG_XNACK_MASK)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_XNACK_MASK) ; encoding: [0x16,0xf8,0x02,0xb9]
// SICIVI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// SICIVI-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_XNACK_MASK)
// SICIVI-ERR-NEXT: {{^}}                       ^
// GFX9-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX11-ERR: :[[@LINE-6]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU

s_getreg_b32 s2, hwreg(HW_REG_POPS_PACKER)
// GFX10:    s_getreg_b32 s2, hwreg(HW_REG_POPS_PACKER) ; encoding: [0x19,0xf8,0x02,0xb9]
// SICIVI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// SICIVI-ERR-NEXT: {{^}}s_getreg_b32 s2, hwreg(HW_REG_POPS_PACKER)
// SICIVI-ERR-NEXT: {{^}}                       ^
// GFX9-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU
// GFX11-ERR: :[[@LINE-6]]:{{[0-9]+}}: error: invalid hardware register: not supported on this GPU

s_cmpk_le_u32 s2, -1
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GCN-NEXT: {{^}}s_cmpk_le_u32 s2, -1
// GCN-NEXT: {{^}}                  ^

s_cmpk_le_u32 s2, 0x1ffff
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GCN-NEXT: {{^}}s_cmpk_le_u32 s2, 0x1ffff
// GCN-NEXT: {{^}}                  ^

s_cmpk_le_u32 s2, 0x10000
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GCN-NEXT: {{^}}s_cmpk_le_u32 s2, 0x10000
// GCN-NEXT: {{^}}                  ^

s_mulk_i32 s2, 0xFFFFFFFFFFFF0000
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GCN-NEXT: {{^}}s_mulk_i32 s2, 0xFFFFFFFFFFFF0000
// GCN-NEXT: {{^}}               ^

s_mulk_i32 s2, 0x10000
// GCN: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GCN-NEXT: {{^}}s_mulk_i32 s2, 0x10000
// GCN-NEXT: {{^}}               ^
