// RUN: llvm-mc -arch=amdgcn -show-encoding -mcpu=gfx1200 %s | FileCheck --check-prefix=GFX12 %s

s_add_i32 s0, s1, s2
// GFX12: s_add_co_i32 s0, s1, s2                 ; encoding: [0x01,0x02,0x00,0x81]

s_add_u32 s0, s1, s2
// GFX12: s_add_co_u32 s0, s1, s2                 ; encoding: [0x01,0x02,0x00,0x80]

s_add_u64 s[0:1], s[2:3], s[4:5]
// GFX12: s_add_nc_u64 s[0:1], s[2:3], s[4:5]     ; encoding: [0x02,0x04,0x80,0xa9]

s_addc_u32 s0, s1, s2
// GFX12: s_add_co_ci_u32 s0, s1, s2              ; encoding: [0x01,0x02,0x00,0x82]

s_sub_i32 s0, s1, s2
// GFX12: s_sub_co_i32 s0, s1, s2                 ; encoding: [0x01,0x02,0x80,0x81]

s_sub_u32 s0, s1, s2
// GFX12: s_sub_co_u32 s0, s1, s2                 ; encoding: [0x01,0x02,0x80,0x80]

s_sub_u64 s[0:1], s[2:3], s[4:5]
// GFX12: s_sub_nc_u64 s[0:1], s[2:3], s[4:5]     ; encoding: [0x02,0x04,0x00,0xaa]

s_subb_u32 s0, s1, s2
// GFX12: s_sub_co_ci_u32 s0, s1, s2              ; encoding: [0x01,0x02,0x80,0x82]

s_min_f32 s5, s1, s2
// GFX12: s_min_num_f32 s5, s1, s2                ; encoding: [0x01,0x02,0x05,0xa1]

s_max_f32 s5, s1, s2
// GFX12: s_max_num_f32 s5, s1, s2                ; encoding: [0x01,0x02,0x85,0xa1]

s_max_f16 s5, s1, s2
// GFX12: s_max_num_f16 s5, s1, s2                ; encoding: [0x01,0x02,0x05,0xa6]

s_min_f16 s5, s1, s2
// GFX12: s_min_num_f16 s5, s1, s2                ; encoding: [0x01,0x02,0x85,0xa5]
