// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

v_cvt_pk_f16_bf8 v1, v2 clamp
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 clamp
// GFX1210-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_bf8 v1, v2 mul:2
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 mul:2
// GFX1210-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_bf8 v1, v2 row_share:1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 row_share:1
// GFX1210-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 clamp
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1210-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 clamp
// GFX1210-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 mul:2
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 mul:2
// GFX1210-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 row_share:1
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1210-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 row_share:1
// GFX1210-ERR-NEXT:{{^}}                        ^
