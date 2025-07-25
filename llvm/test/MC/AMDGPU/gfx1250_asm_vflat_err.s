// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: --strict-whitespace %s

global_load_b96 v[1:3], v[0:1], off
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

flat_load_b32 v5, v[2:3] scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}flat_load_b32 v5, v[2:3] scale_offset
// GFX1250-ERR-NEXT:{{^}}                         ^

flat_load_b32 v5, v[2:3] offset:32 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}flat_load_b32 v5, v[2:3] offset:32 scale_offset
// GFX1250-ERR-NEXT:{{^}}                                   ^

flat_store_b32 v[2:3], v5 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}flat_store_b32 v[2:3], v5 scale_offset
// GFX1250-ERR-NEXT:{{^}}                          ^

flat_atomic_add v[2:3], v2 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}flat_atomic_add v[2:3], v2 scale_offset
// GFX1250-ERR-NEXT:{{^}}                           ^

global_load_b32 v5, v[2:3], off offset:32 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}global_load_b32 v5, v[2:3], off offset:32 scale_offset
// GFX1250-ERR-NEXT:{{^}}                                          ^

global_store_b32 v[2:3], v5, off offset:32 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}global_store_b32 v[2:3], v5, off offset:32 scale_offset
// GFX1250-ERR-NEXT:{{^}}                                           ^

global_atomic_add v[2:3], v2, off scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}global_atomic_add v[2:3], v2, off scale_offset
// GFX1250-ERR-NEXT:{{^}}                                  ^

global_load_addtid_b32 v5, s[2:3] scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}global_load_addtid_b32 v5, s[2:3] scale_offset
// GFX1250-ERR-NEXT:{{^}}                                  ^

global_store_addtid_b32 v5, s[2:3] scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}global_store_addtid_b32 v5, s[2:3] scale_offset
// GFX1250-ERR-NEXT:{{^}}                                   ^

scratch_load_b32 v5, off, s1 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}scratch_load_b32 v5, off, s1 scale_offset
// GFX1250-ERR-NEXT:{{^}}                             ^

scratch_load_b32 v5, off, off offset:32 scale_offset
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1250-ERR-NEXT:{{^}}scratch_load_b32 v5, off, off offset:32 scale_offset
// GFX1250-ERR-NEXT:{{^}}                                        ^
