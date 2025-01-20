// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX13-ERR --implicit-check-not=error: --strict-whitespace %s

v_exclusive_scan_and_b32 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_and_b32 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_and_b32 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_and_b32 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_and_b32 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_and_b32 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_and_b32 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_and_b32 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_and_b32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_and_b32_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_max_i16 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_i16 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_max_i16 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_i16 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_max_i16 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_i16 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_max_i16 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_i16 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_max_i16_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_max_i16_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_max_i32 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_i32 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_max_i32 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_i32 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_max_i32 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_i32 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_max_i32 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_i32 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_max_i32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_max_i32_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_max_u16 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_u16 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_max_u16 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_u16 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_max_u16 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_u16 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_max_u16 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_u16 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_max_u16_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_max_u16_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_max_u32 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_u32 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_max_u32 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_u32 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_max_u32 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_u32 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_max_u32 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_max_u32 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_max_u32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_max_u32_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_min_i16 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_i16 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_min_i16 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_i16 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_min_i16 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_i16 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_min_i16 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_i16 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_min_i16_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_min_i16_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_min_i32 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_i32 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_min_i32 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_i32 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_min_i32 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_i32 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_min_i32 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_i32 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_min_i32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_min_i32_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_min_u16 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_u16 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_min_u16 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_u16 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_min_u16 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_u16 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_min_u16 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_u16 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_min_u16_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_min_u16_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_min_u32 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_u32 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_min_u32 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_u32 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_min_u32 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_u32 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_min_u32 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_min_u32 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_min_u32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_min_u32_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_or_b32 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_or_b32 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                ^

v_exclusive_scan_or_b32 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_or_b32 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                   ^

v_exclusive_scan_or_b32 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_or_b32 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                   ^

v_exclusive_scan_or_b32 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_or_b32 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                ^

v_exclusive_scan_or_b32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_or_b32_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_sum_i32 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_sum_i32 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_sum_i32 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_sum_i32 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_sum_i32 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_sum_i32 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_sum_i32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_sum_i32_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_sum_u32 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_sum_u32 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_sum_u32 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_sum_u32 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_sum_u32 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_sum_u32 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_sum_u32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_sum_u32_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_xor_b32 v5, v3, -v1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_xor_b32 v5, v3, -v1
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_xor_b32 v5, v3, v1 clamp
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_xor_b32 v5, v3, v1 clamp
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_xor_b32 v5, v3, v1 mul:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_xor_b32 v5, v3, v1 mul:2
// GFX13-ERR-NEXT:{{^}}                                    ^

v_exclusive_scan_xor_b32 v5, v3, |v1|
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX13-ERR-NEXT:{{^}}v_exclusive_scan_xor_b32 v5, v3, |v1|
// GFX13-ERR-NEXT:{{^}}                                 ^

v_exclusive_scan_xor_b32_dpp v5, v3, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_exclusive_scan_xor_b32_dpp v5, v3, v1 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: dpp variant of this instruction is not supported

v_cvt_i32_f64_e64_dpp v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_i32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_i32_f64_e64_dpp v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_i32_e64_dpp v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_i32_e64_dpp v[4:5], v2 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_i32_e64_dpp v[4:5], v2 quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_f64_e64_dpp v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f32_f64_e64_dpp v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_f32_e64_dpp v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_f32_e64_dpp v[4:5], v2 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_f32_e64_dpp v[4:5], v2 quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_u32_f64_e64_dpp v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_u32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_u32_f64_e64_dpp v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_u32_e64_dpp v[4:5], v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_u32_e64_dpp v[4:5], v2 row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_cvt_f64_u32_e64_dpp v[4:5], v2 quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_trunc_f64_e64_dpp v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_trunc_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_trunc_f64_e64_dpp v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ceil_f64_e64_dpp v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ceil_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_ceil_f64_e64_dpp v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rndne_f64_e64_dpp v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rndne_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rndne_f64_e64_dpp v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_floor_f64_e64_dpp v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_floor_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_floor_f64_e64_dpp v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_exp_i32_f64_e64_dpp v2, v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_exp_i32_f64_e64_dpp v2, v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_exp_i32_f64_e64_dpp v2, v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_mant_f64_e64_dpp v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_mant_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_frexp_mant_f64_e64_dpp v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fract_f64_e64_dpp v[2:3], v[4:5] dpp8:[7,6,5,4,3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fract_f64_e64_dpp v[2:3], v[4:5] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fract_f64_e64_dpp v[2:3], v[4:5] quad_perm:[3,2,1,0]
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rcp_f64_e64_dpp v[4:5], v[2:3] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_rsq_f64_e64_dpp v[4:5], v[2:3] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_sqrt_f64_e64_dpp v[4:5], v[2:3] row_share:1
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
