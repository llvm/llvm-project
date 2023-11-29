// RUN: llvm-mc -arch=amdgcn -show-encoding -mcpu=gfx1200 %s | FileCheck --check-prefix=GFX12 %s

s_add_i32 s0, s1, s2
// GFX12: encoding: [0x01,0x02,0x00,0x81]

s_add_u32 s0, s1, s2
// GFX12: encoding: [0x01,0x02,0x00,0x80]

s_addc_u32 s0, s1, s2
// GFX12: encoding: [0x01,0x02,0x00,0x82]

s_sub_i32 s0, s1, s2
// GFX12: encoding: [0x01,0x02,0x80,0x81]

s_sub_u32 s0, s1, s2
// GFX12: encoding: [0x01,0x02,0x80,0x80]

s_subb_u32 s0, s1, s2
// GFX12: encoding: [0x01,0x02,0x80,0x82]
