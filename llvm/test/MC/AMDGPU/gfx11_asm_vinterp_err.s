// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck %s -check-prefix=GFX11-ERR --implicit-check-not=error: --strict-whitespace

//===----------------------------------------------------------------------===//
// VINTERP src operands must be VGPRs.
// Check that other operand kinds are rejected by assembler.
//===----------------------------------------------------------------------===//

v_interp_p10_f32 v0, s1, v2, v3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p10_f32 v0, v1, s2, v3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p10_f32 v0, v1, v2, s3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p2_f32 v0, 1, v2, v3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p2_f32 v0, v1, 2, v3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p2_f32 v0, v1, v2, 3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p10_f16_f32 v0, s1, v2, v3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p10_f16_f32 v0, v1, s2, v3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p10_f16_f32 v0, v1, v2, s3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p2_f16_f32 v0, 1, v2, v3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p2_f16_f32 v0, v1, 2, v3
// GFX11-ERR: error: invalid operand for instruction

v_interp_p2_f16_f32 v0, v1, v2, 3
// GFX11-ERR: error: invalid operand for instruction
