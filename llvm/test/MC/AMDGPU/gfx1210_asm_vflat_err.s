// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

flat_load_b32 v5, v[2:3] scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}flat_load_b32 v5, v[2:3] scale_offset
// GFX1210-ERR-NEXT:{{^}}                         ^

flat_load_b32 v5, v[2:3] offset:32 scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}flat_load_b32 v5, v[2:3] offset:32 scale_offset
// GFX1210-ERR-NEXT:{{^}}                                   ^

flat_store_b32 v[2:3], v5 scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}flat_store_b32 v[2:3], v5 scale_offset
// GFX1210-ERR-NEXT:{{^}}                          ^

flat_atomic_add v[1:2], v2 scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}flat_atomic_add v[1:2], v2 scale_offset
// GFX1210-ERR-NEXT:{{^}}                           ^

global_load_b32 v5, v[2:3], off offset:32 scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}global_load_b32 v5, v[2:3], off offset:32 scale_offset
// GFX1210-ERR-NEXT:{{^}}                                          ^

global_store_b32 v[2:3], v5, off offset:32 scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}global_store_b32 v[2:3], v5, off offset:32 scale_offset
// GFX1210-ERR-NEXT:{{^}}                                           ^

global_atomic_add v[1:2], v2, off scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}global_atomic_add v[1:2], v2, off scale_offset
// GFX1210-ERR-NEXT:{{^}}                                  ^

global_load_addtid_b32 v5, s[2:3] scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}global_load_addtid_b32 v5, s[2:3] scale_offset
// GFX1210-ERR-NEXT:{{^}}                                  ^

global_store_addtid_b32 v5, s[2:3] scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}global_store_addtid_b32 v5, s[2:3] scale_offset
// GFX1210-ERR-NEXT:{{^}}                                   ^

scratch_load_b32 v5, off, s1 scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}scratch_load_b32 v5, off, s1 scale_offset
// GFX1210-ERR-NEXT:{{^}}                             ^

scratch_load_b32 v5, off, off offset:32 scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}scratch_load_b32 v5, off, off offset:32 scale_offset
// GFX1210-ERR-NEXT:{{^}}                                        ^
