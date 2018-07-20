// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

v_interp_p1_f32 v1, v0, attr0.x
// SI: v_interp_p1_f32 v1, v0, attr0.x ; encoding: [0x00,0x00,0x04,0xc8]
// VI: v_interp_p1_f32_e32 v1, v0, attr0.x ; encoding: [0x00,0x00,0x04,0xd4]

v_interp_p1_f32 v2, v0, attr0.y
// SI: v_interp_p1_f32 v2, v0, attr0.y ; encoding: [0x00,0x01,0x08,0xc8]
// VI: v_interp_p1_f32_e32 v2, v0, attr0.y ; encoding: [0x00,0x01,0x08,0xd4]

v_interp_p1_f32 v3, v0, attr0.z
// SI: v_interp_p1_f32 v3, v0, attr0.z ; encoding: [0x00,0x02,0x0c,0xc8]
// VI: v_interp_p1_f32_e32 v3, v0, attr0.z ; encoding: [0x00,0x02,0x0c,0xd4]

v_interp_p1_f32 v4, v0, attr0.w
// SI: v_interp_p1_f32 v4, v0, attr0.w ; encoding: [0x00,0x03,0x10,0xc8]
// VI: v_interp_p1_f32_e32 v4, v0, attr0.w ; encoding: [0x00,0x03,0x10,0xd4]

v_interp_p1_f32 v5, v0, attr0.x
// SI: v_interp_p1_f32 v5, v0, attr0.x ; encoding: [0x00,0x00,0x14,0xc8]
// VI: v_interp_p1_f32_e32 v5, v0, attr0.x ; encoding: [0x00,0x00,0x14,0xd4]

v_interp_p1_f32 v6, v0, attr1.x
// SI: v_interp_p1_f32 v6, v0, attr1.x ; encoding: [0x00,0x04,0x18,0xc8]
// VI: v_interp_p1_f32_e32 v6, v0, attr1.x ; encoding: [0x00,0x04,0x18,0xd4]

v_interp_p1_f32 v7, v0, attr2.y
// SI: v_interp_p1_f32 v7, v0, attr2.y ; encoding: [0x00,0x09,0x1c,0xc8]
// VI: v_interp_p1_f32_e32 v7, v0, attr2.y ; encoding: [0x00,0x09,0x1c,0xd4]

v_interp_p1_f32 v8, v0, attr3.z
// SI: v_interp_p1_f32 v8, v0, attr3.z ; encoding: [0x00,0x0e,0x20,0xc8]
// VI: v_interp_p1_f32_e32 v8, v0, attr3.z ; encoding: [0x00,0x0e,0x20,0xd4]

v_interp_p1_f32 v9, v0, attr4.w
// SI: v_interp_p1_f32 v9, v0, attr4.w ; encoding: [0x00,0x13,0x24,0xc8]
// VI: v_interp_p1_f32_e32 v9, v0, attr4.w ; encoding: [0x00,0x13,0x24,0xd4]

v_interp_p1_f32 v10, v0, attr63.w
// SI: v_interp_p1_f32 v10, v0, attr63.w ; encoding: [0x00,0xff,0x28,0xc8]
// VI: v_interp_p1_f32_e32 v10, v0, attr63.w ; encoding: [0x00,0xff,0x28,0xd4]


v_interp_p2_f32 v2, v1, attr0.x
// SI: v_interp_p2_f32 v2, v1, attr0.x ; encoding: [0x01,0x00,0x09,0xc8]
// VI: v_interp_p2_f32_e32 v2, v1, attr0.x ; encoding: [0x01,0x00,0x09,0xd4]

v_interp_p2_f32 v3, v1, attr0.y
// SI: v_interp_p2_f32 v3, v1, attr0.y ; encoding: [0x01,0x01,0x0d,0xc8]
// VI: v_interp_p2_f32_e32 v3, v1, attr0.y ; encoding: [0x01,0x01,0x0d,0xd4]

v_interp_p2_f32 v4, v1, attr0.z
// SI: v_interp_p2_f32 v4, v1, attr0.z ; encoding: [0x01,0x02,0x11,0xc8]
// VI: v_interp_p2_f32_e32 v4, v1, attr0.z ; encoding: [0x01,0x02,0x11,0xd4]

v_interp_p2_f32 v5, v1, attr0.w
// SI: v_interp_p2_f32 v5, v1, attr0.w ; encoding: [0x01,0x03,0x15,0xc8]
// VI: v_interp_p2_f32_e32 v5, v1, attr0.w ; encoding: [0x01,0x03,0x15,0xd4]

v_interp_p2_f32 v6, v1, attr0.x
// SI: v_interp_p2_f32 v6, v1, attr0.x ; encoding: [0x01,0x00,0x19,0xc8]
// VI: v_interp_p2_f32_e32 v6, v1, attr0.x ; encoding: [0x01,0x00,0x19,0xd4]

v_interp_p2_f32 v7, v1, attr1.x
// SI: v_interp_p2_f32 v7, v1, attr1.x ; encoding: [0x01,0x04,0x1d,0xc8]
// VI: v_interp_p2_f32_e32 v7, v1, attr1.x ; encoding: [0x01,0x04,0x1d,0xd4]

v_interp_p2_f32 v8, v1, attr63.x
// SI: v_interp_p2_f32 v8, v1, attr63.x ; encoding: [0x01,0xfc,0x21,0xc8]
// VI: v_interp_p2_f32_e32 v8, v1, attr63.x ; encoding: [0x01,0xfc,0x21,0xd4]


v_interp_mov_f32 v0, p10, attr0.x
// SI: v_interp_mov_f32 v0, p10, attr0.x ; encoding: [0x00,0x00,0x02,0xc8]
// VI: v_interp_mov_f32_e32 v0, p10, attr0.x ; encoding: [0x00,0x00,0x02,0xd4]

v_interp_mov_f32 v1, p20, attr0.x
// SI: v_interp_mov_f32 v1, p20, attr0.x ; encoding: [0x01,0x00,0x06,0xc8]
// VI: v_interp_mov_f32_e32 v1, p20, attr0.x ; encoding: [0x01,0x00,0x06,0xd4]

v_interp_mov_f32 v2, p0, attr0.x
// SI: v_interp_mov_f32 v2, p0, attr0.x ; encoding: [0x02,0x00,0x0a,0xc8]
// VI: v_interp_mov_f32_e32 v2, p0, attr0.x ; encoding: [0x02,0x00,0x0a,0xd4]

v_interp_mov_f32 v4, p10, attr0.y
// SI: v_interp_mov_f32 v4, p10, attr0.y ; encoding: [0x00,0x01,0x12,0xc8]
// VI: v_interp_mov_f32_e32 v4, p10, attr0.y ; encoding: [0x00,0x01,0x12,0xd4]

v_interp_mov_f32 v5, p10, attr0.z
// SI: v_interp_mov_f32 v5, p10, attr0.z ; encoding: [0x00,0x02,0x16,0xc8]
// VI: v_interp_mov_f32_e32 v5, p10, attr0.z ; encoding: [0x00,0x02,0x16,0xd4]

v_interp_mov_f32 v6, p10, attr0.w
// SI: v_interp_mov_f32 v6, p10, attr0.w ; encoding: [0x00,0x03,0x1a,0xc8]
// VI: v_interp_mov_f32_e32 v6, p10, attr0.w ; encoding: [0x00,0x03,0x1a,0xd4]

v_interp_mov_f32 v7, p10, attr0.x
// SI: v_interp_mov_f32 v7, p10, attr0.x ; encoding: [0x00,0x00,0x1e,0xc8]
// VI: v_interp_mov_f32_e32 v7, p10, attr0.x ; encoding: [0x00,0x00,0x1e,0xd4]

v_interp_mov_f32 v9, p10, attr63.y
// SI: v_interp_mov_f32 v9, p10, attr63.y ; encoding: [0x00,0xfd,0x26,0xc8]
// VI: v_interp_mov_f32_e32 v9, p10, attr63.y ; encoding: [0x00,0xfd,0x26,0xd4]

