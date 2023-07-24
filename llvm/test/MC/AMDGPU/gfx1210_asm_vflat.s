// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefix=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

global_load_b32 v0, v[2:3], off nv
// GFX1210: global_load_b32 v0, v[2:3], off nv      ; encoding: [0xfc,0x00,0x05,0xee,0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_load_b32 v0, v[2:3], off nv
// GFX12-ERR-NEXT:{{^}}                                ^

global_store_b32 v[2:3], v0, off nv
// GFX1210: global_store_b32 v[2:3], v0, off nv     ; encoding: [0xfc,0x80,0x06,0xee,0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_store_b32 v[2:3], v0, off nv
// GFX12-ERR-NEXT:{{^}}                                 ^

global_atomic_add v[1:2], v2, off nv
// GFX1210: global_atomic_add_u32 v[1:2], v2, off nv ; encoding: [0xfc,0x40,0x0d,0xee,0x00,0x00,0x00,0x01,0x01,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_atomic_add v[1:2], v2, off nv
// GFX12-ERR-NEXT:{{^}}                                  ^

global_load_addtid_b32 v5, s[2:3] nv
// GFX1210: global_load_addtid_b32 v5, s[2:3] nv    ; encoding: [0x82,0x00,0x0a,0xee,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_load_addtid_b32 v5, s[2:3] nv
// GFX12-ERR-NEXT:{{^}}                                  ^

scratch_load_b32 v0, v2, off nv
// GFX1210: scratch_load_b32 v0, v2, off nv         ; encoding: [0xfc,0x00,0x05,0xed,0x00,0x00,0x02,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_load_b32 v0, v2, off nv
// GFX12-ERR-NEXT:{{^}}                             ^

scratch_store_b32 v2, v0, off nv
// GFX1210: scratch_store_b32 v2, v0, off nv        ; encoding: [0xfc,0x80,0x06,0xed,0x00,0x00,0x02,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_store_b32 v2, v0, off nv
// GFX12-ERR-NEXT:{{^}}                              ^

flat_load_b32 v0, v[2:3] nv
// GFX1210: flat_load_b32 v0, v[2:3] nv             ; encoding: [0xfc,0x00,0x05,0xec,0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}flat_load_b32 v0, v[2:3] nv
// GFX12-ERR-NEXT:{{^}}                         ^

flat_store_b32 v[2:3], v0 nv
// GFX1210: flat_store_b32 v[2:3], v0 nv            ; encoding: [0xfc,0x80,0x06,0xec,0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}flat_store_b32 v[2:3], v0 nv
// GFX12-ERR-NEXT:{{^}}                          ^

flat_atomic_add v[1:2], v2 nv
// GFX1210: flat_atomic_add_u32 v[1:2], v2 nv       ; encoding: [0xfc,0x40,0x0d,0xec,0x00,0x00,0x00,0x01,0x01,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}flat_atomic_add v[1:2], v2 nv
// GFX12-ERR-NEXT:{{^}}                           ^
