// RUN: llvm-mc -triple=amdgcn -mcpu=gfx908 -show-encoding %s | FileCheck --check-prefix=CHECK-MI %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx90a -show-encoding %s | FileCheck --check-prefix=CHECK-MI %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx942 -show-encoding %s | FileCheck --check-prefix=CHECK-MI %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx950 -show-encoding %s | FileCheck --check-prefix=CHECK-MI %s

v_pk_fmac_f16 v5, v1, v2
// CHECK-MI: [0x01,0x05,0x0a,0x78]

v_pk_fmac_f16 v5, v1, v2 quad_perm:[0,1,2,3]
// CHECK-MI: [0xfa,0x04,0x0a,0x78,0x01,0xe4,0x00,0xff]

v_pk_fmac_f16 v5, v1, v2 quad_perm:[0,1,2,3] row_mask:0x0 bank_mask:0x0
// CHECK-MI: [0xfa,0x04,0x0a,0x78,0x01,0xe4,0x00,0x00]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:BYTE_0 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x00,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x01,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:BYTE_2 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x02,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:BYTE_3 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x03,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x04,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x05,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_SEXT src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x0e,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x16,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x16,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x00,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x01,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x02,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x03,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x04,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x05,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x00]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x01]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x02]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x03]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x04]

v_pk_fmac_f16_sdwa v5, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x05]

v_pk_fmac_f16_sdwa v5, v1, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x06]

v_pk_fmac_f16_sdwa v5, v1, -v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
// CHECK-MI: [0xf9,0x04,0x0a,0x78,0x01,0x06,0x06,0x16]
