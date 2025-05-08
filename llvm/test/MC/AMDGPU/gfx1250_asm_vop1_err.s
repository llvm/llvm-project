// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_cvt_pk_f16_bf8 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_bf8 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_bf8 v1, v2 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_bf8 v1, v2 row_share:1
// GFX1250-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 clamp
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 clamp
// GFX1250-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 mul:2
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 mul:2
// GFX1250-ERR-NEXT:{{^}}                        ^

v_cvt_pk_f16_fp8 v1, v2 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// GFX1250-ERR-NEXT:{{^}}v_cvt_pk_f16_fp8 v1, v2 row_share:1
// GFX1250-ERR-NEXT:{{^}}                        ^
