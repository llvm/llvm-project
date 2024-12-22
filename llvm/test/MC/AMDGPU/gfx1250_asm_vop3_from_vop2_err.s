// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

v_fmaak_f32_e64_dpp v4, v2, v6, 3 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fmamk_f32_e64_dpp v4, v2, 3, v6 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fmaak_f16_e64_dpp v4, v2, v6, 3 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported

v_fmamk_f16_e64_dpp v4, v2, 3, v6 row_share:1
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: e64_dpp variant of this instruction is not supported
