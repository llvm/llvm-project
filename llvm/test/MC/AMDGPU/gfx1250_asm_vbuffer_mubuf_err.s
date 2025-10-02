// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

buffer_load_b32 v5, v1, s[8:11], s3 offen offset:4095 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}buffer_load_b32 v5, v1, s[8:11], s3 offen offset:4095 scale_offset
// GFX1250-ERR-NEXT:{{^}}                                                      ^
