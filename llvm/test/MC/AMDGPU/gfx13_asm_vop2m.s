// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding %s | FileCheck -strict-whitespace --check-prefix=GFX13 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding %s 2>&1 | FileCheck -strict-whitespace -implicit-check-not=error: -check-prefix=GFX13-ERR %s

v_mov_2src_b64 v5, v2, v3
// GFX13: v_mov_2src_b64 v5, v2, v3               ; encoding: [0x05,0x00,0x84,0x6e,0x02,0x61,0x20,0x00]

v_mov_2src_b64 v5, v2, v3 aux_data:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_bpermute_b32 v5, v2, v3
// GFX13: v_bpermute_b32 v5, v2, v3               ; encoding: [0x05,0x00,0x86,0x6e,0x02,0x61,0x20,0x00]

v_bpermute_b32 v5, v2, v3 aux_data:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_permute_pair_bcast_b32 v5, v2
// GFX13: v_permute_pair_bcast_b32 v5, v2         ; encoding: [0x05,0x00,0x88,0x6e,0x02,0x01,0x00,0x00]

v_permute_pair_bcast_b32 v5, v2 aux_data:2
// GFX13: v_permute_pair_bcast_b32 v5, v2 aux_data:2 ; encoding: [0x05,0x00,0x88,0x6e,0x02,0x01,0x00,0x08]

v_permute_pair_bcast_b32 v5, v2, v3
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_permute_pair_gensgpr_b32 v5, v2, s[4:5]
// GFX13: v_permute_pair_gensgpr_b32 v5, v2, s[4:5] ; encoding: [0x05,0x00,0x8a,0x6e,0x02,0x81,0x00,0x00]

v_permute_pair_gensgpr_b32 v5, v2, s[4:5] aux_data:2
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.

v_permute_pair_gensgpr_b32 v5, v2, v3
// GFX13-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction

v_permute_pair_2src_rotate_group_b32 v5, v2, v3
// GFX13: v_permute_pair_2src_rotate_group_b32 v5, v2, v3 ; encoding: [0x05,0x00,0x8c,0x6e,0x02,0x61,0x20,0x00]

v_permute_pair_2src_rotate_group_b32 v5, v2, v3 aux_data:2
// GFX13: v_permute_pair_2src_rotate_group_b32 v5, v2, v3 aux_data:2 ; encoding: [0x05,0x00,0x8c,0x6e,0x02,0x61,0x20,0x08]

