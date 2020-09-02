// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=GFX9-ERR --implicit-check-not=error: %s

v_addc_co_u32_e32 v3, vcc, 12345, v3, vcc
// GFX9-ERR: error: invalid operand (violates constant bus restrictions)

v_cndmask_b32 v0, 12345, v1, vcc
// GFX9-ERR: error: invalid operand (violates constant bus restrictions)
