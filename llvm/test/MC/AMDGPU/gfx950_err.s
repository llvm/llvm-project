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

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], v6 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], v6 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], v6 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_2xpk16_fp6_f32 v[20:25], v[10:25], v[10:25], v6 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], v6 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], v6 mul:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], v6 div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand
v_cvt_scalef32_2xpk16_bf6_f32 v[20:25], v[10:25], v[10:25], v6 clamp div:2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:4095 glc

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:4095 slc

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:4095 dlc

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
buffer_atomic_pk_add_bf16 v5, off, s[8:11], s3 offset:4095 glc slc dlc

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_maximum3_f16 v0, v1, v2, v3

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_minimum3_f16 v0, v1, v2, v3

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_maximum_f16 v0, v1, v2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_minimum_f16 v0, v1, v2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_maximum_f32 v0, v1, v2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_minimum_f32 v0, v1, v2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
v_maximum3_f32 v0, s1, s2, v3

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
v_maximum3_f32 v0, v3, s1, s2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
v_maximum3_f32 v0, s1, v3, s2

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
v_minimum3_f32 v0, s1, s2, v3

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: literal operands are not supported
v_minimum3_f32 v0, v1, v2, 0xdeadbeef

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
v_pk_minimum3_f16 v0, s1, s2, v3

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
v_pk_maximum3_f16 v0, s1, s2, v3

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_sr_f16_f32 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_sr_bf16_f32 v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_bf8_bf16 v0, v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_bf8_f16 v0, v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_bf8_f32 v0, v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_fp8_bf16 v0, v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_fp8_f16 v0, v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_fp8_f32 v0, v1, v2, v3 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_pk32_bf6_bf16 v[0:5], v[0:15], v16, v17 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_pk32_bf6_f16 v[0:5], v[6:21], v22, v23 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_pk32_fp6_bf16 v[0:5], v[6:21], v22, v23 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_pk32_fp6_f16 v[0:5], v[6:21], v22, v23 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_pk32_bf6_f32 v[0:5], v[6:37], v38, v39 clamp

// GFX950: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cvt_scalef32_sr_pk32_fp6_f32 v[0:5], v[6:37], v38, v39 clamp
