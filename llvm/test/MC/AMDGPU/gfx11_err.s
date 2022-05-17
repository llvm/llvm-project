// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefix=GFX11 --implicit-check-not=error: %s

s_delay_alu
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction

s_delay_alu instid9(VALU_DEP_1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid field name instid9

s_delay_alu instid0(VALU_DEP_9)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: invalid value name VALU_DEP_9

s_delay_alu instid0(1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: expected a value name

s_delay_alu instid0(VALU_DEP_9|)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: expected a right parenthesis

s_delay_alu instid0(VALU_DEP_1) | (SALU_CYCLE_1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: expected a field name

s_delay_alu instid0(VALU_DEP_1) | SALU_CYCLE_1)
// GFX11: [[@LINE-1]]:{{[0-9]+}}: error: expected a left parenthesis
