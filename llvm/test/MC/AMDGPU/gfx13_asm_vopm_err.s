// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX13-ERR --implicit-check-not=error: --strict-whitespace %s
v_wmma_bf16_16x16_bf16 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_bf16_16x16_bf16 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                   ^

v_wmma_f16_16x16_bf8_bf8 v[904:907], 0, v[902:903], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_bf8 v[904:907], 0, v[902:903], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f16_16x16_bf8_bf8 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_bf8 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f16_16x16_bf8_fp8 v[904:907], 0, v[902:903], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_fp8 v[904:907], 0, v[902:903], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f16_16x16_bf8_fp8 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_fp8 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f16_16x16_f16 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_f16 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_f16_16x16_fp8_bf8 v[904:907], 0, v[902:903], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_bf8 v[904:907], 0, v[902:903], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f16_16x16_fp8_bf8 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_bf8 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f16_16x16_fp8_fp8 v[904:907], 0, v[902:903], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_fp8 v[904:907], 0, v[902:903], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f16_16x16_fp8_fp8 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_fp8 v[908:911], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f32_16x16_bf16 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf16 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                  ^

v_wmma_f32_16x16_bf8_bf8 v[904:911], 0, v[902:903], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_bf8 v[904:911], 0, v[902:903], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f32_16x16_bf8_bf8 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_bf8 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f32_16x16_bf8_fp8 v[904:911], 0, v[902:903], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_fp8 v[904:911], 0, v[902:903], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f32_16x16_bf8_fp8 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_fp8 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f32_16x16_f16 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_f16 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_f32_16x16_f8f6f4 v[916:923], 0, v[908:915], 0, v924, v925 aux_data:1152 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_f8f6f4 v[916:923], 0, v[908:915], 0, v924, v925 aux_data:1152 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_wmma_f32_16x16_fp8_bf8 v[904:911], 0, v[902:903], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_bf8 v[904:911], 0, v[902:903], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f32_16x16_fp8_bf8 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_bf8 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f32_16x16_fp8_fp8 v[904:911], 0, v[902:903], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_fp8 v[904:911], 0, v[902:903], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f32_16x16_fp8_fp8 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_fp8 v[908:915], 0, v[904:907], 0 clamp
// GFX13-ERR-NEXT:{{^}}                                     ^

v_wmma_f32_16x16_iu4 v[902:909], 0, v901, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu4 v[902:909], 0, v901, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_f32_16x16_iu4 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu4 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_f32_16x16_iu4 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu4 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_f32_16x16_iu8 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu8 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_f32_16x16_iu8 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu8 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_f32i32_16x16_iu4 v[902:909], 0, v901, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu4 v[902:909], 0, v901, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_wmma_f32i32_16x16_iu4 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu4 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_wmma_f32i32_16x16_iu4 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu4 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_wmma_f32i32_16x16_iu8 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu8 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_wmma_f32i32_16x16_iu8 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu8 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_wmma_i32_16x16_iu4 v[902:909], 0, v901, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu4 v[902:909], 0, v901, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_i32_16x16_iu4 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu4 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_i32_16x16_iu4 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu4 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_i32_16x16_iu8 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu8 v[904:911], 0, v[902:903], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_i32_16x16_iu8 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu8 v[908:915], 0, v[904:907], 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                 ^

v_wmma_bf16_16x16_bf16 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_bf16_16x16_bf16 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                               ^

v_wmma_f16_16x16_bf8_bf8 v[904:907], v[900:901], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_bf8 v[904:907], v[900:901], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f16_16x16_bf8_bf8 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_bf8 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f16_16x16_bf8_fp8 v[904:907], v[900:901], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_fp8 v[904:907], v[900:901], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f16_16x16_bf8_fp8 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_fp8 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f16_16x16_f16 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_f16 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_f16_16x16_fp8_bf8 v[904:907], v[900:901], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_bf8 v[904:907], v[900:901], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f16_16x16_fp8_bf8 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_bf8 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f16_16x16_fp8_fp8 v[904:907], v[900:901], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_fp8 v[904:907], v[900:901], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f16_16x16_fp8_fp8 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_fp8 v[908:911], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f32_16x16_bf16 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf16 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                              ^

v_wmma_f32_16x16_bf8_bf8 v[904:911], v[900:901], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_bf8 v[904:911], v[900:901], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f32_16x16_bf8_bf8 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_bf8 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f32_16x16_bf8_fp8 v[904:911], v[900:901], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_fp8 v[904:911], v[900:901], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f32_16x16_bf8_fp8 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_fp8 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f32_16x16_f16 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_f16 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_f32_16x16_f8f6f4 v[916:923], v[900:907], 0, 0, v924, v925 aux_data:1152 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_f8f6f4 v[916:923], v[900:907], 0, 0, v924, v925 aux_data:1152 clamp
// GFX13-ERR-NEXT:{{^}}                                                ^

v_wmma_f32_16x16_fp8_bf8 v[904:911], v[900:901], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_bf8 v[904:911], v[900:901], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f32_16x16_fp8_bf8 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_bf8 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f32_16x16_fp8_fp8 v[904:911], v[900:901], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_fp8 v[904:911], v[900:901], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f32_16x16_fp8_fp8 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_fp8 v[908:915], v[900:903], 0, 0 clamp
// GFX13-ERR-NEXT:{{^}}                                                 ^

v_wmma_f32_16x16_iu4 v[902:909], v900, 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu4 v[902:909], v900, 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                       ^

v_wmma_f32_16x16_iu4 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu4 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_f32_16x16_iu4 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu4 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_f32_16x16_iu8 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu8 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_f32_16x16_iu8 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu8 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_f32i32_16x16_iu4 v[902:909], v900, 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu4 v[902:909], v900, 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                          ^

v_wmma_f32i32_16x16_iu4 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu4 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                ^

v_wmma_f32i32_16x16_iu4 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu4 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                ^

v_wmma_f32i32_16x16_iu8 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu8 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                ^

v_wmma_f32i32_16x16_iu8 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu8 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                ^

v_wmma_i32_16x16_iu4 v[902:909], v900, 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu4 v[902:909], v900, 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                       ^

v_wmma_i32_16x16_iu4 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu4 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_i32_16x16_iu4 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu4 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_i32_16x16_iu8 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu8 v[904:911], v[900:901], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_i32_16x16_iu8 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu8 v[908:915], v[900:903], 0, 0 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_f32_16x16_fp8_fp8 v[904:911], v[900:901], v[902:903], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_fp8 v[904:911], v[900:901], v[902:903], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f16_16x16_fp8_fp8 v[904:907], v[900:901], v[902:903], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_fp8 v[904:907], v[900:901], v[902:903], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f32_16x16_fp8_bf8 v[904:911], v[900:901], v[902:903], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_bf8 v[904:911], v[900:901], v[902:903], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f16_16x16_fp8_bf8 v[904:907], v[900:901], v[902:903], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_bf8 v[904:907], v[900:901], v[902:903], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f32_16x16_bf8_fp8 v[904:911], v[900:901], v[902:903], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_fp8 v[904:911], v[900:901], v[902:903], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f16_16x16_bf8_fp8 v[904:907], v[900:901], v[902:903], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_fp8 v[904:907], v[900:901], v[902:903], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f32_16x16_bf8_bf8 v[904:911], v[900:901], v[902:903], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_bf8 v[904:911], v[900:901], v[902:903], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f16_16x16_bf8_bf8 v[904:907], v[900:901], v[902:903], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_bf8 v[904:907], v[900:901], v[902:903], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f32_16x16_f16 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_f16 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f16_16x16_f16 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_f16 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f32_16x16_bf16 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf16 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                          ^

v_wmma_bf16_16x16_bf16 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_bf16_16x16_bf16 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                           ^

v_wmma_f32_16x16_fp8_fp8 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_fp8 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f16_16x16_fp8_fp8 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_fp8 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f32_16x16_fp8_bf8 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_fp8_bf8 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f16_16x16_fp8_bf8 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_fp8_bf8 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f32_16x16_bf8_fp8 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_fp8 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f16_16x16_bf8_fp8 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_fp8 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f32_16x16_bf8_bf8 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_bf8_bf8 v[908:915], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f16_16x16_bf8_bf8 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f16_16x16_bf8_bf8 v[908:911], v[900:903], v[904:907], 1 clamp
// GFX13-ERR-NEXT:{{^}}                                                             ^

v_wmma_f32_16x16_iu8 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu8 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f32i32_16x16_iu8 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu8 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                            ^

v_wmma_i32_16x16_iu8 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu8 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f32_16x16_iu4 v[902:909], v900, v901, 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu4 v[902:909], v900, v901, 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_f32i32_16x16_iu4 v[902:909], v900, v901, 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu4 v[902:909], v900, v901, 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                ^

v_wmma_i32_16x16_iu4 v[902:909], v900, v901, 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu4 v[902:909], v900, v901, 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                             ^

v_wmma_f32_16x16_iu8 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu8 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f32i32_16x16_iu8 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu8 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                            ^

v_wmma_i32_16x16_iu8 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu8 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f32_16x16_iu4 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu4 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f32i32_16x16_iu4 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu4 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                            ^

v_wmma_i32_16x16_iu4 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu4 v[904:911], v[900:901], v[902:903], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f32_16x16_iu4 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_iu4 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f32i32_16x16_iu4 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32i32_16x16_iu4 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                            ^

v_wmma_i32_16x16_iu4 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_i32_16x16_iu4 v[908:915], v[900:903], v[904:907], 1 signed_a signed_b clamp
// GFX13-ERR-NEXT:{{^}}                                                         ^

v_wmma_f32_16x16_f8f6f4 v[916:923], v[900:907], v[908:915], 1, v924, v925 aux_data:1152 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: only zero is supported as immediate operand
// GFX13-ERR-NEXT:{{^}}v_wmma_f32_16x16_f8f6f4 v[916:923], v[900:907], v[908:915], 1, v924, v925 aux_data:1152 clamp
// GFX13-ERR-NEXT:{{^}}                                                            ^
