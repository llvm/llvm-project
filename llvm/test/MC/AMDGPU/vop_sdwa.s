// RUN: llvm-mc -arch=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=CIVI --check-prefix=VI
// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSI --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSI --check-prefix=NOSICI
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOSICI

// ToDo: converters
// ToDo: VOPC
// ToDo: VOP2b (see vop_dpp.s)
// ToDo: V_MAC_F32 (see vop_dpp.s)
// ToDo: sext()
// ToDo: intrinsics


// NOSICI: error:
// VI: v_mov_b32_sdwa v1, v2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x7e,0x02,0x10,0x06,0x06]
v_mov_b32 v1, v2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD

// NOSICI: error:
// VI: v_mov_b32_sdwa v3, v4 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 ; encoding: [0xf9,0x02,0x06,0x7e,0x04,0x11,0x05,0x06]
v_mov_b32 v3, v4 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1

// NOSICI: error:
// VI: v_mov_b32_sdwa v15, v99 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:WORD_0 ; encoding: [0xf9,0x02,0x1e,0x7e,0x63,0x0a,0x04,0x06]
v_mov_b32 v15, v99 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:WORD_0

// NOSICI: error:
// VI: v_min_u32_sdwa v194, v13, v1 dst_sel:BYTE_3 dst_unused:UNUSED_SEXT src0_sel:BYTE_3 src1_sel:BYTE_2 ; encoding: [0xf9,0x02,0x84,0x1d,0x0d,0x0b,0x03,0x02]
v_min_u32 v194, v13, v1 dst_sel:BYTE_3 dst_unused:UNUSED_SEXT src0_sel:BYTE_3 src1_sel:BYTE_2

// NOSICI: error:
// VI: v_min_u32_sdwa v255, v4, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:WORD_1 ; encoding: [0xf9,0x02,0xfe,0x1d,0x04,0x04,0x02,0x05]
v_min_u32 v255, v4, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:WORD_1

// NOSICI: error:
// VI: v_min_u32_sdwa v200, v200, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD ; encoding: [0xf9,0x02,0x90,0x1d,0xc8,0x05,0x01,0x06]
v_min_u32 v200, v200, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD

// NOSICI: error:
// VI: v_min_u32_sdwa v1, v1, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x1c,0x01,0x06,0x00,0x06]
v_min_u32 v1, v1, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
