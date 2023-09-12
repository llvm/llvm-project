// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=GFX9ERR --implicit-check-not=error: %s

v_cvt_f16_u16_e64 v5, 0.5
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_cvt_f16_u16_e64 v5, -4.0
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_add_u16_e64 v5, v1, 0.5
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_add_u16_e64 v5, v1, -4.0
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_cvt_f16_i16_e64 v5, 0.5
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_cvt_f16_i16_e64 v5, -4.0
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_add_u16_e64 v5, 0.5, v2
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_add_u16_e64 v5, -4.0, v2
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_subrev_u16_e64 v5, v1, 0.5
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_subrev_u16_e64 v5, v1, -4.0
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: literal operands are not supported

v_cvt_u32_f64 v5, v[0:1] quad_perm:[0,2,1,1] row_mask:0xf bank_mask:0xf
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

global_load_lds_dword v[2:3], off
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

global_load_dword v[2:3], off
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction

scratch_load_dword v2, off, offset:256
// GFX9ERR: :[[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
