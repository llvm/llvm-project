// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_cvt_pk_f16_bf8 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_bf8 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                        ^

v_cvt_f32_bf16 v5, v1 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_cvt_f32_bf16 v5, v1 clamp
// GFX1250-ERR-NEXT:{{^}}                      ^

v_cvt_f32_bf16 v5, v1 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_f32_bf16 v5, v1 mul:2
// GFX1250-ERR-NEXT:{{^}}                      ^

v_cvt_f32_bf16 v5, v1 div:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_f32_bf16 v5, v1 div:2
// GFX1250-ERR-NEXT:{{^}}                      ^

v_cos_bf16 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_cos_bf16 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                  ^

v_cos_bf16 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cos_bf16 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                  ^

v_exp_bf16 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_exp_bf16 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                  ^

v_exp_bf16 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_exp_bf16 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                  ^

v_log_bf16 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_log_bf16 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                  ^

v_log_bf16 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_log_bf16 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                  ^

v_rcp_bf16 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_rcp_bf16 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                  ^

v_rcp_bf16 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_rcp_bf16 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                  ^

v_rsq_bf16 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_rsq_bf16 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                  ^

v_rsq_bf16 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_rsq_bf16 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                  ^

v_sin_bf16 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_sin_bf16 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                  ^

v_sin_bf16 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_sin_bf16 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                  ^

v_sqrt_bf16 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_sqrt_bf16 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                   ^

v_sqrt_bf16 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_sqrt_bf16 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                   ^

v_tanh_bf16 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_tanh_bf16 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                   ^

v_tanh_bf16 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_tanh_bf16 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                   ^
