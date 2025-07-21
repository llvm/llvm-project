// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX12-ERR --implicit-check-not=error: --strict-whitespace %s

global_load_b32 v0, v[2:3], off nv
// GFX1250: global_load_b32 v0, v[2:3], off nv      ; encoding: [0xfc,0x00,0x05,0xee,0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_load_b32 v0, v[2:3], off nv
// GFX12-ERR-NEXT:{{^}}                                ^

global_store_b32 v[2:3], v0, off nv
// GFX1250: global_store_b32 v[2:3], v0, off nv     ; encoding: [0xfc,0x80,0x06,0xee,0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_store_b32 v[2:3], v0, off nv
// GFX12-ERR-NEXT:{{^}}                                 ^

global_atomic_add v[2:3], v2, off nv
// GFX1250: global_atomic_add_u32 v[2:3], v2, off nv ; encoding: [0xfc,0x40,0x0d,0xee,0x00,0x00,0x00,0x01,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_atomic_add v[2:3], v2, off nv
// GFX12-ERR-NEXT:{{^}}                                  ^

global_load_addtid_b32 v5, s[2:3] nv
// GFX1250: global_load_addtid_b32 v5, s[2:3] nv    ; encoding: [0x82,0x00,0x0a,0xee,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_load_addtid_b32 v5, s[2:3] nv
// GFX12-ERR-NEXT:{{^}}                                  ^

scratch_load_b32 v0, v2, off nv
// GFX1250: scratch_load_b32 v0, v2, off nv         ; encoding: [0xfc,0x00,0x05,0xed,0x00,0x00,0x02,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_load_b32 v0, v2, off nv
// GFX12-ERR-NEXT:{{^}}                             ^

scratch_store_b32 v2, v0, off nv
// GFX1250: scratch_store_b32 v2, v0, off nv        ; encoding: [0xfc,0x80,0x06,0xed,0x00,0x00,0x02,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_store_b32 v2, v0, off nv
// GFX12-ERR-NEXT:{{^}}                              ^

flat_load_b32 v0, v[2:3] nv
// GFX1250: flat_load_b32 v0, v[2:3] nv             ; encoding: [0xfc,0x00,0x05,0xec,0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}flat_load_b32 v0, v[2:3] nv
// GFX12-ERR-NEXT:{{^}}                         ^

flat_store_b32 v[2:3], v0 nv
// GFX1250: flat_store_b32 v[2:3], v0 nv            ; encoding: [0xfc,0x80,0x06,0xec,0x00,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}flat_store_b32 v[2:3], v0 nv
// GFX12-ERR-NEXT:{{^}}                          ^

flat_atomic_add v[2:3], v2 nv
// GFX1250: flat_atomic_add_u32 v[2:3], v2 nv       ; encoding: [0xfc,0x40,0x0d,0xec,0x00,0x00,0x00,0x01,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}flat_atomic_add v[2:3], v2 nv
// GFX12-ERR-NEXT:{{^}}                           ^

scratch_load_b32 v5, v2, off nv
// GFX1250: scratch_load_b32 v5, v2, off nv         ; encoding: [0xfc,0x00,0x05,0xed,0x05,0x00,0x02,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: nv is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_load_b32 v5, v2, off nv
// GFX12-ERR-NEXT:{{^}}                             ^

global_load_b32 v5, v1, s[2:3] offset:32 scale_offset
// GFX1250: global_load_b32 v5, v1, s[2:3] offset:32 scale_offset ; encoding: [0x02,0x00,0x05,0xee,0x05,0x00,0x01,0x00,0x01,0x20,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_load_b32 v5, v1, s[2:3] offset:32 scale_offset
// GFX12-ERR-NEXT:{{^}}                                         ^

global_store_b32 v5, v1, s[2:3] offset:32 scale_offset
// GFX1250: global_store_b32 v5, v1, s[2:3] offset:32 scale_offset ; encoding: [0x02,0x80,0x06,0xee,0x00,0x00,0x81,0x00,0x05,0x20,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_store_b32 v5, v1, s[2:3] offset:32 scale_offset
// GFX12-ERR-NEXT:{{^}}                                          ^

global_atomic_add_u32 v2, v5, s[2:3] scale_offset
// GFX1250: global_atomic_add_u32 v2, v5, s[2:3] scale_offset ; encoding: [0x02,0x40,0x0d,0xee,0x00,0x00,0x81,0x02,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}global_atomic_add_u32 v2, v5, s[2:3] scale_offset
// GFX12-ERR-NEXT:{{^}}                                     ^

scratch_load_b32 v5, v2, off scale_offset
// GFX1250: scratch_load_b32 v5, v2, off scale_offset ; encoding: [0x7c,0x00,0x05,0xed,0x05,0x00,0x03,0x00,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_load_b32 v5, v2, off scale_offset
// GFX12-ERR-NEXT:{{^}}                             ^

scratch_load_b32 v5, v2, off offset:32 scale_offset
// GFX1250: scratch_load_b32 v5, v2, off offset:32 scale_offset ; encoding: [0x7c,0x00,0x05,0xed,0x05,0x00,0x03,0x00,0x02,0x20,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_load_b32 v5, v2, off offset:32 scale_offset
// GFX12-ERR-NEXT:{{^}}                                       ^

scratch_load_b32 v5, v2, s1 offset:32 scale_offset
// GFX1250: scratch_load_b32 v5, v2, s1 offset:32 scale_offset ; encoding: [0x01,0x00,0x05,0xed,0x05,0x00,0x03,0x00,0x02,0x20,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_load_b32 v5, v2, s1 offset:32 scale_offset
// GFX12-ERR-NEXT:{{^}}                                      ^

scratch_store_b32 v2, v5, off scale_offset
// GFX1250: scratch_store_b32 v2, v5, off scale_offset ; encoding: [0x7c,0x80,0x06,0xed,0x00,0x00,0x83,0x02,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_store_b32 v2, v5, off scale_offset
// GFX12-ERR-NEXT:{{^}}                              ^

scratch_store_b32 v2, v5, s1 scale_offset
// GFX1250: scratch_store_b32 v2, v5, s1 scale_offset ; encoding: [0x01,0x80,0x06,0xed,0x00,0x00,0x83,0x02,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: scale_offset is not supported on this GPU
// GFX12-ERR-NEXT:{{^}}scratch_store_b32 v2, v5, s1 scale_offset
// GFX12-ERR-NEXT:{{^}}                             ^

tensor_save s[0:1]
// GFX1250: tensor_save s[0:1] ; encoding: [0x00,0x80,0x1b,0xee,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_save s[0:1] th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1250: tensor_save s[0:1] th:TH_STORE_BYPASS scope:SCOPE_SYS ; encoding: [0x00,0x80,0x1b,0xee,0x00,0x00,0x3c,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_save s[0:1] offset:32
// GFX1250: tensor_save s[0:1] offset:32 ; encoding: [0x00,0x80,0x1b,0xee,0x00,0x00,0x00,0x00,0x00,0x20,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_stop
// GFX1250: tensor_stop ; encoding: [0x7c,0xc0,0x1b,0xee,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

tensor_stop th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1250: tensor_stop th:TH_STORE_BYPASS scope:SCOPE_SYS ; encoding: [0x7c,0xc0,0x1b,0xee,0x00,0x00,0x3c,0x00,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

flat_atomic_add_f32 v1, v2, s[2:3] offset:8000000 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_add_f32 v1, v2, s[2:3] offset:8000000 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x80,0x15,0xec,0x00,0x00,0x11,0x01,0x01,0x00,0x12,0x7a]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_add_f32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_add_f32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x15,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_add_u32 v1, v2, s[2:3] offset:-64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_add_u32 v1, v2, s[2:3] offset:-64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x40,0x0d,0xec,0x00,0x00,0x11,0x01,0x01,0xc0,0xff,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_add_u32 v2, v3, s[2:3] offset:-64
// GFX1250: flat_atomic_add_u32 v2, v3, s[2:3] offset:-64 ; encoding: [0x02,0x40,0x0d,0xec,0x00,0x00,0x80,0x01,0x02,0xc0,0xff,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_add_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_add_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0xc0,0x10,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_add_u64 v[0:1], v2, v[2:3], s[2:3] offset:-64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_add_u64 v[0:1], v2, v[2:3], s[2:3] offset:-64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0xc0,0x10,0xec,0x00,0x00,0x11,0x01,0x02,0xc0,0xff,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_and_b32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_and_b32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x00,0x0f,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_and_b32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_and_b32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x00,0x0f,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_and_b64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_and_b64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x40,0x12,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_and_b64 v[0:1], v2, v[2:3], s[2:3] offset:-64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_and_b64 v[0:1], v2, v[2:3], s[2:3] offset:-64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x40,0x12,0xec,0x00,0x00,0x11,0x01,0x02,0xc0,0xff,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_cmpswap_b32 v0, v2, v[2:3], s[2:3] scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_cmpswap_b32 v0, v2, v[2:3], s[2:3] scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x00,0x0d,0xec,0x00,0x00,0x11,0x01,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_cmpswap_b32 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_cmpswap_b32 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x00,0x0d,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_cmpswap_b64 v[0:1], v2, v[2:5], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_cmpswap_b64 v[0:1], v2, v[2:5], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x80,0x10,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_cmpswap_b64 v2, v[2:5], s[2:3]
// GFX1250: flat_atomic_cmpswap_b64 v2, v[2:5], s[2:3] ; encoding: [0x02,0x80,0x10,0xec,0x00,0x00,0x00,0x01,0x02,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_cond_sub_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_cond_sub_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x00,0x14,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_cond_sub_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_cond_sub_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x00,0x14,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_dec_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_dec_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x00,0x10,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_dec_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_dec_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x00,0x10,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_dec_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_dec_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x40,0x13,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_dec_u64 v[0:1], v2, v[2:3], s[2:3] offset:-64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_dec_u64 v[0:1], v2, v[2:3], s[2:3] offset:-64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x40,0x13,0xec,0x00,0x00,0x11,0x01,0x02,0xc0,0xff,0xff]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_inc_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_inc_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0xc0,0x0f,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_inc_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_inc_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0xc0,0x0f,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_inc_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_inc_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x00,0x13,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_inc_u64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_inc_u64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x00,0x13,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_max_num_f32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_max_num_f32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x80,0x14,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_num_f32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_max_num_f32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x14,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_i32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_max_i32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x80,0x0e,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_i32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_max_i32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x0e,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_i64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_max_i64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0xc0,0x11,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_i64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_max_i64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0xc0,0x11,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_max_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_max_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0xc0,0x0e,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_max_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0xc0,0x0e,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_max_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x00,0x12,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_max_u64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_max_u64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x00,0x12,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_min_num_f32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_min_num_f32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x40,0x14,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_num_f32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_min_num_f32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x40,0x14,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_i32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_min_i32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x00,0x0e,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_i32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_min_i32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x00,0x0e,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_i64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_min_i64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x40,0x11,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_i64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_min_i64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x40,0x11,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_min_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_min_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x40,0x0e,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_min_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x40,0x0e,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_min_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x80,0x11,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_min_u64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_min_u64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x80,0x11,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_or_b32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_or_b32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x40,0x0f,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_or_b32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_or_b32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x40,0x0f,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_or_b64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_or_b64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x80,0x12,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_or_b64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_or_b64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x80,0x12,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_sub_clamp_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_sub_clamp_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0xc0,0x0d,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_sub_clamp_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_sub_clamp_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0xc0,0x0d,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_sub_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_sub_u32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x80,0x0d,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_sub_u32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_sub_u32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x0d,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_sub_u64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_sub_u64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x00,0x11,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_sub_u64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_sub_u64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x00,0x11,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_swap_b32 v0, v2, s[2:3] scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_swap_b32 v0, v2, s[2:3] scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0xc0,0x0c,0xec,0x00,0x00,0x11,0x01,0x00,0x00,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_swap_b32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_swap_b32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0xc0,0x0c,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_swap_b64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_swap_b64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0x40,0x10,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_swap_b64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_swap_b64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x40,0x10,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_xor_b32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_xor_b32 v1, v2, s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x80,0x0f,0xec,0x00,0x00,0x11,0x01,0x01,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_xor_b32 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_xor_b32 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x0f,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_xor_b64 v2, v[2:3], s[2:3] offset:64
// GFX1250: flat_atomic_xor_b64 v2, v[2:3], s[2:3] offset:64 ; encoding: [0x02,0xc0,0x12,0xec,0x00,0x00,0x00,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_xor_b64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_xor_b64 v[0:1], v2, v[2:3], s[2:3] offset:64 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0xc0,0x12,0xec,0x00,0x00,0x11,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_atomic_pk_add_f16 v1, v2, s[2:3] offset:8000000 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_pk_add_f16 v1, v2, s[2:3] offset:8000000 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x40,0x16,0xec,0x00,0x00,0x11,0x01,0x01,0x00,0x12,0x7a]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_pk_add_f16 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_pk_add_f16 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x40,0x16,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_pk_add_bf16 v1, v2, s[2:3] offset:8000000 scale_offset th:TH_ATOMIC_RETURN
// GFX1250: flat_atomic_pk_add_bf16 v1, v2, s[2:3] offset:8000000 scale_offset th:TH_ATOMIC_RETURN ; encoding: [0x02,0x80,0x16,0xec,0x00,0x00,0x11,0x01,0x01,0x00,0x12,0x7a]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_atomic_pk_add_bf16 v2, v3, s[2:3] offset:64
// GFX1250: flat_atomic_pk_add_bf16 v2, v3, s[2:3] offset:64 ; encoding: [0x02,0x80,0x16,0xec,0x00,0x00,0x80,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

flat_load_b128 v[2:5], v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_b128 v[2:5], v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0xc0,0x05,0xec,0x02,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_b32 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_b32 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x00,0x05,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_b64 v[2:3], v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_b64 v[2:3], v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x40,0x05,0xec,0x02,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_b96 v[2:4], v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_b96 v[2:4], v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x80,0x05,0xec,0x02,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_d16_b16 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_d16_b16 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x00,0x08,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_d16_hi_b16 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_d16_hi_b16 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0xc0,0x08,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_d16_hi_i8 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_d16_hi_i8 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x80,0x08,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_d16_hi_u8 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_d16_hi_u8 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x40,0x08,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_d16_i8 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_d16_i8 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0xc0,0x07,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_d16_u8 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_d16_u8 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x80,0x07,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_i16 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_i16 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0xc0,0x04,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_i8 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_i8 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x40,0x04,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_u16 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_u16 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x80,0x04,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_u8 v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_u8 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x00,0x04,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_load_dword v1, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_load_b32 v1, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x00,0x05,0xec,0x01,0x00,0x01,0x00,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_store_b128 v2, v[2:5], s[2:3] offset:64 scale_offset
// GFX1250: flat_store_b128 v2, v[2:5], s[2:3] offset:64 scale_offset ; encoding: [0x02,0x40,0x07,0xec,0x00,0x00,0x01,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_store_b16 v2, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_store_b16 v2, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x40,0x06,0xec,0x00,0x00,0x01,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_store_b32 v2, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_store_b32 v2, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x80,0x06,0xec,0x00,0x00,0x01,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_store_b64 v2, v[2:3], s[2:3] offset:64 scale_offset
// GFX1250: flat_store_b64 v2, v[2:3], s[2:3] offset:64 scale_offset ; encoding: [0x02,0xc0,0x06,0xec,0x00,0x00,0x01,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_store_b8 v2, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_store_b8 v2, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x00,0x06,0xec,0x00,0x00,0x01,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_store_b96 v2, v[2:4], s[2:3] offset:64 scale_offset
// GFX1250: flat_store_b96 v2, v[2:4], s[2:3] offset:64 scale_offset ; encoding: [0x02,0x00,0x07,0xec,0x00,0x00,0x01,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_store_d16_hi_b16 v2, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_store_d16_hi_b16 v2, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x40,0x09,0xec,0x00,0x00,0x01,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.

flat_store_d16_hi_b8 v2, v2, s[2:3] offset:64 scale_offset
// GFX1250: flat_store_d16_hi_b8 v2, v2, s[2:3] offset:64 scale_offset ; encoding: [0x02,0x00,0x09,0xec,0x00,0x00,0x01,0x01,0x02,0x40,0x00,0x00]
// GFX12-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: not a valid operand.
