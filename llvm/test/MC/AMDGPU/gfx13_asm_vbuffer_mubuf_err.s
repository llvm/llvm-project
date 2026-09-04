// RUN: not llvm-mc -triple=amdgpu13.10 -show-encoding < %s 2>&1 | FileCheck --check-prefix=GFX13-ERR --implicit-check-not=error: %s

buffer_load_b32 v5, off, rsrcidx(s[8:11]), s3 offset:0
// GFX13-ERR: :[[@LINE-1]]:34: error: rsrcidx operand must be a 32-bit SGPR or VGPR

buffer_load_b32 v5, off, rsrcidx(v[8:11]), s3 offset:0
// GFX13-ERR: :[[@LINE-1]]:34: error: rsrcidx operand must be a 32-bit SGPR or VGPR

buffer_load_b32 v5, off, rsrcidx s8, s3 offset:0
// GFX13-ERR: :[[@LINE-1]]:34: error: expected left paren after rsrcidx

buffer_load_b32 v5, off, rsrcidx(s8, s3 offset:0
// GFX13-ERR: :[[@LINE-1]]:36: error: expected closing parenthesis

buffer_load_b32 v5, off, rsrcidx(off), s3 offset:0
// GFX13-ERR: :[[@LINE-1]]:34: error: invalid register name
