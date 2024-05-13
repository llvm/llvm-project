// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx950 %s 2>&1 | FileCheck --check-prefix=GFX950 --implicit-check-not=error: %s

// GFX950: :[[@LINE+1]]:27: error: invalid operand for instruction
v_permlane16_swap_b32 v0, s0

// GFX950: :[[@LINE+1]]:27: error: invalid operand for instruction
v_permlane16_swap_b32 v0, m0

// GFX950: :[[@LINE+1]]:27: error: invalid operand for instruction
v_permlane16_swap_b32 v0, vcc

// GFX950: :[[@LINE+1]]:27: error: invalid operand for instruction
v_permlane16_swap_b32 v0, vcc_lo

// GFX950: :[[@LINE+1]]:23: error: invalid operand for instruction
v_permlane16_swap_b32 s0, v0

// GFX950: :[[@LINE+1]]:34: error: invalid operand for instruction
v_permlane16_swap_b32_e32 v1, v2 bound_ctrl:1

// GFX950: :[[@LINE+1]]:34: error: invalid operand for instruction
v_permlane16_swap_b32_e32 v1, v2 bound_ctrl:0

// GFX950: :[[@LINE+1]]:34: error: invalid operand for instruction
v_permlane16_swap_b32_e32 v1, v2 fi:1

// GFX950: :[[@LINE+1]]:34: error: invalid operand for instruction
v_permlane16_swap_b32_e32 v1, v2 fi:0

// GFX950: :[[@LINE+1]]:34: error: invalid operand for instruction
v_permlane16_swap_b32_e32 v1, v2 bound_ctrl:1 fi:1

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_fp8_f16 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp8_f16 v1, v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp8_f16 v1, v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp8_f16 v1, v2, v3 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_fp8_bf16 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp8_bf16 v1, v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp8_bf16 v1, v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp8_bf16 v1, v2, v3 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_bf8_f16 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf8_f16 v1, v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf8_f16 v1, v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf8_f16 v1, v2, v3 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_bf8_bf16 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf8_bf16 v1, v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf8_bf16 v1, v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf8_bf16 v1, v2, v3 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f32_fp4 v[2:3], v2, v3 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp4_f32 v1, v1, v2, v3 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f16_fp4 v1, v2, v3 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf16_fp4 v1, v2, v3 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_f32_fp6 v[2:33], v[2:7], v6 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f32_fp6 v[2:33], v[2:7], v6 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f32_fp6 v[2:33], v[2:7], v6 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f32_fp6 v[2:33], v[2:7], v6 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_f32_bf6 v[2:33], v[2:7], v6 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f32_bf6 v[2:33], v[2:7], v6 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f32_bf6 v[2:33], v[2:7], v6 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f32_bf6 v[2:33], v[2:7], v6 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_bf16_bf6 v[10:25], v[20:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_bf16_bf6 v[10:25], v[20:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_bf16_bf6 v[10:25], v[20:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_bf16_bf6 v[10:25], v[20:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_f16_bf6 v[10:25], v[20:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f16_bf6 v[10:25], v[20:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f16_bf6 v[10:25], v[20:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f16_bf6 v[10:25], v[20:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_bf16_fp6 v[10:25], v[20:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_bf16_fp6 v[10:25], v[20:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_bf16_fp6 v[10:25], v[20:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_bf16_fp6 v[10:25], v[20:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_f16_fp6 v[10:25], v[20:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f16_fp6 v[10:25], v[20:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f16_fp6 v[10:25], v[20:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_f16_fp6 v[10:25], v[20:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_bf6_f16 v[20:25], v[10:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_fp6_bf16 v[20:25], v[10:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk32_fp6_f16 v[20:25], v[10:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_f16_fp8 v[20:25], v[10:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f16_fp8 v[20:25], v[10:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f16_fp8 v[20:25], v[10:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f16_fp8 v[20:25], v[10:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_f16_bf8 v[20:25], v[10:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f16_bf8 v[20:25], v[10:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f16_bf8 v[20:25], v[10:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_f16_bf8 v[20:25], v[10:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_bf16_fp8 v[20:25], v[10:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf16_fp8 v[20:25], v[10:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf16_fp8 v[20:25], v[10:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf16_fp8 v[20:25], v[10:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_bf16_bf8 v[20:25], v[10:25], v8 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf16_bf8 v[20:25], v[10:25], v8 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf16_bf8 v[20:25], v[10:25], v8 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_bf16_bf8 v[20:25], v[10:25], v8 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_fp4_f16 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp4_f16 v1, v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp4_f16 v1, v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp4_f16 v1, v2, v3 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_pk_fp4_bf16 v1, v2, v3 clamp div:2
