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
