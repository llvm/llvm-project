// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s  | FileCheck %s
// RUN: llvm-mc -triple x86_64-unknown-unknown < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-WARN

// CHECK-NO-WARN-NOT: warning

// CHECK: vmovsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x00,0x10,0xf4]
          vmovsh %xmm28, %xmm29, %xmm30

// CHECK: vmovsh 268435456(%rbp,%r14,8), %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7e,0x0f,0x10,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmovsh 268435456(%rbp,%r14,8), %xmm30 {%k7}

// CHECK: vmovsh (%r9), %xmm30
// CHECK: encoding: [0x62,0x45,0x7e,0x08,0x10,0x31]
          vmovsh (%r9), %xmm30

// CHECK: vmovsh 254(%rcx), %xmm30
// CHECK: encoding: [0x62,0x65,0x7e,0x08,0x10,0x71,0x7f]
          vmovsh 254(%rcx), %xmm30

// CHECK: vmovsh -256(%rdx), %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7e,0x8f,0x10,0x72,0x80]
          vmovsh -256(%rdx), %xmm30 {%k7} {z}

// CHECK: vmovsh %xmm30, 268435456(%rbp,%r14,8) {%k7}
// CHECK: encoding: [0x62,0x25,0x7e,0x0f,0x11,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmovsh %xmm30, 268435456(%rbp,%r14,8) {%k7}

// CHECK: vmovsh %xmm30, (%r9)
// CHECK: encoding: [0x62,0x45,0x7e,0x08,0x11,0x31]
          vmovsh %xmm30, (%r9)

// CHECK: vmovsh %xmm30, 254(%rcx)
// CHECK: encoding: [0x62,0x65,0x7e,0x08,0x11,0x71,0x7f]
          vmovsh %xmm30, 254(%rcx)

// CHECK: vmovsh %xmm30, -256(%rdx) {%k7}
// CHECK: encoding: [0x62,0x65,0x7e,0x0f,0x11,0x72,0x80]
          vmovsh %xmm30, -256(%rdx) {%k7}

// CHECK: vmovw %r12d, %xmm30
// CHECK: encoding: [0x62,0x45,0x7d,0x08,0x6e,0xf4]
          vmovw %r12d, %xmm30

// CHECK: vmovw %xmm30, %r12d
// CHECK: encoding: [0x62,0x45,0x7d,0x08,0x7e,0xf4]
          vmovw %xmm30, %r12d

// CHECK: vmovw 268435456(%rbp,%r14,8), %xmm30
// CHECK: encoding: [0x62,0x25,0x7d,0x08,0x6e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmovw 268435456(%rbp,%r14,8), %xmm30

// CHECK: vmovw (%r9), %xmm30
// CHECK: encoding: [0x62,0x45,0x7d,0x08,0x6e,0x31]
          vmovw (%r9), %xmm30

// CHECK: vmovw 254(%rcx), %xmm30
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x6e,0x71,0x7f]
          vmovw 254(%rcx), %xmm30

// CHECK: vmovw -256(%rdx), %xmm30
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x6e,0x72,0x80]
          vmovw -256(%rdx), %xmm30

// CHECK: vmovw %xmm30, 268435456(%rbp,%r14,8)
// CHECK: encoding: [0x62,0x25,0x7d,0x08,0x7e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmovw %xmm30, 268435456(%rbp,%r14,8)

// CHECK: vmovw %xmm30, (%r9)
// CHECK: encoding: [0x62,0x45,0x7d,0x08,0x7e,0x31]
          vmovw %xmm30, (%r9)

// CHECK: vmovw %xmm30, 254(%rcx)
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x7e,0x71,0x7f]
          vmovw %xmm30, 254(%rcx)

// CHECK: vmovw %xmm30, -256(%rdx)
// CHECK: encoding: [0x62,0x65,0x7d,0x08,0x7e,0x72,0x80]
          vmovw %xmm30, -256(%rdx)

// CHECK: vaddph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x40,0x58,0xf4]
          vaddph %zmm28, %zmm29, %zmm30

// CHECK: vaddph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x10,0x58,0xf4]
          vaddph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vaddph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x14,0x47,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vaddph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x45,0x14,0x50,0x58,0x31]
          vaddph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vaddph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x65,0x14,0x40,0x58,0x71,0x7f]
          vaddph  8128(%rcx), %zmm29, %zmm30

// CHECK: vaddph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x14,0xd7,0x58,0x72,0x80]
          vaddph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vaddsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x00,0x58,0xf4]
          vaddsh %xmm28, %xmm29, %xmm30

// CHECK: vaddsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x10,0x58,0xf4]
          vaddsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vaddsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x16,0x07,0x58,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vaddsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vaddsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x58,0x31]
          vaddsh  (%r9), %xmm29, %xmm30

// CHECK: vaddsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x58,0x71,0x7f]
          vaddsh  254(%rcx), %xmm29, %xmm30

// CHECK: vaddsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x16,0x87,0x58,0x72,0x80]
          vaddsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vcmpneq_usph %zmm28, %zmm29, %k5
// CHECK: encoding: [0x62,0x93,0x14,0x40,0xc2,0xec,0x14]
          vcmpneq_usph %zmm28, %zmm29, %k5

// CHECK: vcmpnlt_uqph {sae}, %zmm28, %zmm29, %k5
// CHECK: encoding: [0x62,0x93,0x14,0x10,0xc2,0xec,0x15]
          vcmpnlt_uqph {sae}, %zmm28, %zmm29, %k5

// CHECK: vcmpnle_uqph 268435456(%rbp,%r14,8), %zmm29, %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x14,0x47,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x16]
          vcmpnle_uqph 268435456(%rbp,%r14,8), %zmm29, %k5 {%k7}

// CHECK: vcmpord_sph (%r9){1to32}, %zmm29, %k5
// CHECK: encoding: [0x62,0xd3,0x14,0x50,0xc2,0x29,0x17]
          vcmpord_sph (%r9){1to32}, %zmm29, %k5

// CHECK: vcmpeq_usph 8128(%rcx), %zmm29, %k5
// CHECK: encoding: [0x62,0xf3,0x14,0x40,0xc2,0x69,0x7f,0x18]
          vcmpeq_usph 8128(%rcx), %zmm29, %k5

// CHECK: vcmpnge_uqph -256(%rdx){1to32}, %zmm29, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x14,0x57,0xc2,0x6a,0x80,0x19]
          vcmpnge_uqph -256(%rdx){1to32}, %zmm29, %k5 {%k7}

// CHECK: vcmpngt_uqsh %xmm28, %xmm29, %k5
// CHECK: encoding: [0x62,0x93,0x16,0x00,0xc2,0xec,0x1a]
          vcmpngt_uqsh %xmm28, %xmm29, %k5

// CHECK: vcmpfalse_ossh {sae}, %xmm28, %xmm29, %k5
// CHECK: encoding: [0x62,0x93,0x16,0x10,0xc2,0xec,0x1b]
          vcmpfalse_ossh {sae}, %xmm28, %xmm29, %k5

// CHECK: vcmpneq_ossh 268435456(%rbp,%r14,8), %xmm29, %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x16,0x07,0xc2,0xac,0xf5,0x00,0x00,0x00,0x10,0x1c]
          vcmpneq_ossh 268435456(%rbp,%r14,8), %xmm29, %k5 {%k7}

// CHECK: vcmpge_oqsh (%r9), %xmm29, %k5
// CHECK: encoding: [0x62,0xd3,0x16,0x00,0xc2,0x29,0x1d]
          vcmpge_oqsh (%r9), %xmm29, %k5

// CHECK: vcmpgt_oqsh 254(%rcx), %xmm29, %k5
// CHECK: encoding: [0x62,0xf3,0x16,0x00,0xc2,0x69,0x7f,0x1e]
          vcmpgt_oqsh 254(%rcx), %xmm29, %k5

// CHECK: vcmptrue_ussh -256(%rdx), %xmm29, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x16,0x07,0xc2,0x6a,0x80,0x1f]
          vcmptrue_ussh -256(%rdx), %xmm29, %k5 {%k7}

// CHECK: vcomish %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x08,0x2f,0xf5]
          vcomish %xmm29, %xmm30

// CHECK: vcomish {sae}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x18,0x2f,0xf5]
          vcomish {sae}, %xmm29, %xmm30

// CHECK: vcomish  268435456(%rbp,%r14,8), %xmm30
// CHECK: encoding: [0x62,0x25,0x7c,0x08,0x2f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcomish  268435456(%rbp,%r14,8), %xmm30

// CHECK: vcomish  (%r9), %xmm30
// CHECK: encoding: [0x62,0x45,0x7c,0x08,0x2f,0x31]
          vcomish  (%r9), %xmm30

// CHECK: vcomish  254(%rcx), %xmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x2f,0x71,0x7f]
          vcomish  254(%rcx), %xmm30

// CHECK: vcomish  -256(%rdx), %xmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x2f,0x72,0x80]
          vcomish  -256(%rdx), %xmm30

// CHECK: vdivph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x40,0x5e,0xf4]
          vdivph %zmm28, %zmm29, %zmm30

// CHECK: vdivph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x10,0x5e,0xf4]
          vdivph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vdivph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x14,0x47,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vdivph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x45,0x14,0x50,0x5e,0x31]
          vdivph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vdivph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x65,0x14,0x40,0x5e,0x71,0x7f]
          vdivph  8128(%rcx), %zmm29, %zmm30

// CHECK: vdivph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x14,0xd7,0x5e,0x72,0x80]
          vdivph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vdivsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x00,0x5e,0xf4]
          vdivsh %xmm28, %xmm29, %xmm30

// CHECK: vdivsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x10,0x5e,0xf4]
          vdivsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vdivsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x16,0x07,0x5e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vdivsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vdivsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x5e,0x31]
          vdivsh  (%r9), %xmm29, %xmm30

// CHECK: vdivsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x5e,0x71,0x7f]
          vdivsh  254(%rcx), %xmm29, %xmm30

// CHECK: vdivsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x16,0x87,0x5e,0x72,0x80]
          vdivsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vmaxph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x40,0x5f,0xf4]
          vmaxph %zmm28, %zmm29, %zmm30

// CHECK: vmaxph {sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x10,0x5f,0xf4]
          vmaxph {sae}, %zmm28, %zmm29, %zmm30

// CHECK: vmaxph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x14,0x47,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vmaxph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x45,0x14,0x50,0x5f,0x31]
          vmaxph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vmaxph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x65,0x14,0x40,0x5f,0x71,0x7f]
          vmaxph  8128(%rcx), %zmm29, %zmm30

// CHECK: vmaxph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x14,0xd7,0x5f,0x72,0x80]
          vmaxph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vmaxsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x00,0x5f,0xf4]
          vmaxsh %xmm28, %xmm29, %xmm30

// CHECK: vmaxsh {sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x10,0x5f,0xf4]
          vmaxsh {sae}, %xmm28, %xmm29, %xmm30

// CHECK: vmaxsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x16,0x07,0x5f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmaxsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vmaxsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x5f,0x31]
          vmaxsh  (%r9), %xmm29, %xmm30

// CHECK: vmaxsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x5f,0x71,0x7f]
          vmaxsh  254(%rcx), %xmm29, %xmm30

// CHECK: vmaxsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x16,0x87,0x5f,0x72,0x80]
          vmaxsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vminph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x40,0x5d,0xf4]
          vminph %zmm28, %zmm29, %zmm30

// CHECK: vminph {sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x10,0x5d,0xf4]
          vminph {sae}, %zmm28, %zmm29, %zmm30

// CHECK: vminph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x14,0x47,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vminph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x45,0x14,0x50,0x5d,0x31]
          vminph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vminph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x65,0x14,0x40,0x5d,0x71,0x7f]
          vminph  8128(%rcx), %zmm29, %zmm30

// CHECK: vminph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x14,0xd7,0x5d,0x72,0x80]
          vminph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vminsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x00,0x5d,0xf4]
          vminsh %xmm28, %xmm29, %xmm30

// CHECK: vminsh {sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x10,0x5d,0xf4]
          vminsh {sae}, %xmm28, %xmm29, %xmm30

// CHECK: vminsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x16,0x07,0x5d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vminsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vminsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x5d,0x31]
          vminsh  (%r9), %xmm29, %xmm30

// CHECK: vminsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x5d,0x71,0x7f]
          vminsh  254(%rcx), %xmm29, %xmm30

// CHECK: vminsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x16,0x87,0x5d,0x72,0x80]
          vminsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vmulph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x40,0x59,0xf4]
          vmulph %zmm28, %zmm29, %zmm30

// CHECK: vmulph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x10,0x59,0xf4]
          vmulph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vmulph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x14,0x47,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vmulph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x45,0x14,0x50,0x59,0x31]
          vmulph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vmulph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x65,0x14,0x40,0x59,0x71,0x7f]
          vmulph  8128(%rcx), %zmm29, %zmm30

// CHECK: vmulph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x14,0xd7,0x59,0x72,0x80]
          vmulph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vmulsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x00,0x59,0xf4]
          vmulsh %xmm28, %xmm29, %xmm30

// CHECK: vmulsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x10,0x59,0xf4]
          vmulsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vmulsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x16,0x07,0x59,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vmulsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vmulsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x59,0x31]
          vmulsh  (%r9), %xmm29, %xmm30

// CHECK: vmulsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x59,0x71,0x7f]
          vmulsh  254(%rcx), %xmm29, %xmm30

// CHECK: vmulsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x16,0x87,0x59,0x72,0x80]
          vmulsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vsubph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x40,0x5c,0xf4]
          vsubph %zmm28, %zmm29, %zmm30

// CHECK: vsubph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x14,0x10,0x5c,0xf4]
          vsubph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vsubph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x14,0x47,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vsubph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x45,0x14,0x50,0x5c,0x31]
          vsubph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vsubph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x65,0x14,0x40,0x5c,0x71,0x7f]
          vsubph  8128(%rcx), %zmm29, %zmm30

// CHECK: vsubph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x14,0xd7,0x5c,0x72,0x80]
          vsubph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vsubsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x00,0x5c,0xf4]
          vsubsh %xmm28, %xmm29, %xmm30

// CHECK: vsubsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x10,0x5c,0xf4]
          vsubsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vsubsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x16,0x07,0x5c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsubsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vsubsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x5c,0x31]
          vsubsh  (%r9), %xmm29, %xmm30

// CHECK: vsubsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x5c,0x71,0x7f]
          vsubsh  254(%rcx), %xmm29, %xmm30

// CHECK: vsubsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x16,0x87,0x5c,0x72,0x80]
          vsubsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vucomish %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x08,0x2e,0xf5]
          vucomish %xmm29, %xmm30

// CHECK: vucomish {sae}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x18,0x2e,0xf5]
          vucomish {sae}, %xmm29, %xmm30

// CHECK: vucomish  268435456(%rbp,%r14,8), %xmm30
// CHECK: encoding: [0x62,0x25,0x7c,0x08,0x2e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vucomish  268435456(%rbp,%r14,8), %xmm30

// CHECK: vucomish  (%r9), %xmm30
// CHECK: encoding: [0x62,0x45,0x7c,0x08,0x2e,0x31]
          vucomish  (%r9), %xmm30

// CHECK: vucomish  254(%rcx), %xmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x2e,0x71,0x7f]
          vucomish  254(%rcx), %xmm30

// CHECK: vucomish  -256(%rdx), %xmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x08,0x2e,0x72,0x80]
          vucomish  -256(%rdx), %xmm30

// CHECK: vcvtdq2ph %zmm29, %ymm30
// CHECK: encoding: [0x62,0x05,0x7c,0x48,0x5b,0xf5]
          vcvtdq2ph %zmm29, %ymm30

// CHECK: vcvtdq2ph {rn-sae}, %zmm29, %ymm30
// CHECK: encoding: [0x62,0x05,0x7c,0x18,0x5b,0xf5]
          vcvtdq2ph {rn-sae}, %zmm29, %ymm30

// CHECK: vcvtdq2ph  268435456(%rbp,%r14,8), %ymm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7c,0x4f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtdq2ph  268435456(%rbp,%r14,8), %ymm30 {%k7}

// CHECK: vcvtdq2ph  (%r9){1to16}, %ymm30
// CHECK: encoding: [0x62,0x45,0x7c,0x58,0x5b,0x31]
          vcvtdq2ph  (%r9){1to16}, %ymm30

// CHECK: vcvtdq2ph  8128(%rcx), %ymm30
// CHECK: encoding: [0x62,0x65,0x7c,0x48,0x5b,0x71,0x7f]
          vcvtdq2ph  8128(%rcx), %ymm30

// CHECK: vcvtdq2ph  -512(%rdx){1to16}, %ymm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7c,0xdf,0x5b,0x72,0x80]
          vcvtdq2ph  -512(%rdx){1to16}, %ymm30 {%k7} {z}

// CHECK: vcvtpd2ph %zmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0xfd,0x48,0x5a,0xf5]
          vcvtpd2ph %zmm29, %xmm30

// CHECK: vcvtpd2ph {rn-sae}, %zmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0xfd,0x18,0x5a,0xf5]
          vcvtpd2ph {rn-sae}, %zmm29, %xmm30

// CHECK: vcvtpd2phz  268435456(%rbp,%r14,8), %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0xfd,0x4f,0x5a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtpd2phz  268435456(%rbp,%r14,8), %xmm30 {%k7}

// CHECK: vcvtpd2ph  (%r9){1to8}, %xmm30
// CHECK: encoding: [0x62,0x45,0xfd,0x58,0x5a,0x31]
          vcvtpd2ph  (%r9){1to8}, %xmm30

// CHECK: vcvtpd2phz  8128(%rcx), %xmm30
// CHECK: encoding: [0x62,0x65,0xfd,0x48,0x5a,0x71,0x7f]
          vcvtpd2phz  8128(%rcx), %xmm30

// CHECK: vcvtpd2ph  -1024(%rdx){1to8}, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0xfd,0xdf,0x5a,0x72,0x80]
          vcvtpd2ph  -1024(%rdx){1to8}, %xmm30 {%k7} {z}

// CHECK: vcvtph2dq %ymm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x48,0x5b,0xf5]
          vcvtph2dq %ymm29, %zmm30

// CHECK: vcvtph2dq {rn-sae}, %ymm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x18,0x5b,0xf5]
          vcvtph2dq {rn-sae}, %ymm29, %zmm30

// CHECK: vcvtph2dq  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7d,0x4f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2dq  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtph2dq  (%r9){1to16}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7d,0x58,0x5b,0x31]
          vcvtph2dq  (%r9){1to16}, %zmm30

// CHECK: vcvtph2dq  4064(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7d,0x48,0x5b,0x71,0x7f]
          vcvtph2dq  4064(%rcx), %zmm30

// CHECK: vcvtph2dq  -256(%rdx){1to16}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7d,0xdf,0x5b,0x72,0x80]
          vcvtph2dq  -256(%rdx){1to16}, %zmm30 {%k7} {z}

// CHECK: vcvtph2pd %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x48,0x5a,0xf5]
          vcvtph2pd %xmm29, %zmm30

// CHECK: vcvtph2pd {sae}, %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x18,0x5a,0xf5]
          vcvtph2pd {sae}, %xmm29, %zmm30

// CHECK: vcvtph2pd  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7c,0x4f,0x5a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2pd  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtph2pd  (%r9){1to8}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7c,0x58,0x5a,0x31]
          vcvtph2pd  (%r9){1to8}, %zmm30

// CHECK: vcvtph2pd  2032(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x48,0x5a,0x71,0x7f]
          vcvtph2pd  2032(%rcx), %zmm30

// CHECK: vcvtph2pd  -256(%rdx){1to8}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7c,0xdf,0x5a,0x72,0x80]
          vcvtph2pd  -256(%rdx){1to8}, %zmm30 {%k7} {z}

// CHECK: vcvtph2psx %ymm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x7d,0x48,0x13,0xf5]
          vcvtph2psx %ymm29, %zmm30

// CHECK: vcvtph2psx {sae}, %ymm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x7d,0x18,0x13,0xf5]
          vcvtph2psx {sae}, %ymm29, %zmm30

// CHECK: vcvtph2psx  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x7d,0x4f,0x13,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2psx  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtph2psx  (%r9){1to16}, %zmm30
// CHECK: encoding: [0x62,0x46,0x7d,0x58,0x13,0x31]
          vcvtph2psx  (%r9){1to16}, %zmm30

// CHECK: vcvtph2psx  4064(%rcx), %zmm30
// CHECK: encoding: [0x62,0x66,0x7d,0x48,0x13,0x71,0x7f]
          vcvtph2psx  4064(%rcx), %zmm30

// CHECK: vcvtph2psx  -256(%rdx){1to16}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x7d,0xdf,0x13,0x72,0x80]
          vcvtph2psx  -256(%rdx){1to16}, %zmm30 {%k7} {z}

// CHECK: vcvtph2qq %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x48,0x7b,0xf5]
          vcvtph2qq %xmm29, %zmm30

// CHECK: vcvtph2qq {rn-sae}, %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x18,0x7b,0xf5]
          vcvtph2qq {rn-sae}, %xmm29, %zmm30

// CHECK: vcvtph2qq  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7d,0x4f,0x7b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2qq  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtph2qq  (%r9){1to8}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7d,0x58,0x7b,0x31]
          vcvtph2qq  (%r9){1to8}, %zmm30

// CHECK: vcvtph2qq  2032(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7d,0x48,0x7b,0x71,0x7f]
          vcvtph2qq  2032(%rcx), %zmm30

// CHECK: vcvtph2qq  -256(%rdx){1to8}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7d,0xdf,0x7b,0x72,0x80]
          vcvtph2qq  -256(%rdx){1to8}, %zmm30 {%k7} {z}

// CHECK: vcvtph2udq %ymm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x48,0x79,0xf5]
          vcvtph2udq %ymm29, %zmm30

// CHECK: vcvtph2udq {rn-sae}, %ymm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x18,0x79,0xf5]
          vcvtph2udq {rn-sae}, %ymm29, %zmm30

// CHECK: vcvtph2udq  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7c,0x4f,0x79,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2udq  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtph2udq  (%r9){1to16}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7c,0x58,0x79,0x31]
          vcvtph2udq  (%r9){1to16}, %zmm30

// CHECK: vcvtph2udq  4064(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x48,0x79,0x71,0x7f]
          vcvtph2udq  4064(%rcx), %zmm30

// CHECK: vcvtph2udq  -256(%rdx){1to16}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7c,0xdf,0x79,0x72,0x80]
          vcvtph2udq  -256(%rdx){1to16}, %zmm30 {%k7} {z}

// CHECK: vcvtph2uqq %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x48,0x79,0xf5]
          vcvtph2uqq %xmm29, %zmm30

// CHECK: vcvtph2uqq {rn-sae}, %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x18,0x79,0xf5]
          vcvtph2uqq {rn-sae}, %xmm29, %zmm30

// CHECK: vcvtph2uqq  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7d,0x4f,0x79,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2uqq  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtph2uqq  (%r9){1to8}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7d,0x58,0x79,0x31]
          vcvtph2uqq  (%r9){1to8}, %zmm30

// CHECK: vcvtph2uqq  2032(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7d,0x48,0x79,0x71,0x7f]
          vcvtph2uqq  2032(%rcx), %zmm30

// CHECK: vcvtph2uqq  -256(%rdx){1to8}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7d,0xdf,0x79,0x72,0x80]
          vcvtph2uqq  -256(%rdx){1to8}, %zmm30 {%k7} {z}

// CHECK: vcvtph2uw %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x48,0x7d,0xf5]
          vcvtph2uw %zmm29, %zmm30

// CHECK: vcvtph2uw {rn-sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x18,0x7d,0xf5]
          vcvtph2uw {rn-sae}, %zmm29, %zmm30

// CHECK: vcvtph2uw  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7c,0x4f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2uw  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtph2uw  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7c,0x58,0x7d,0x31]
          vcvtph2uw  (%r9){1to32}, %zmm30

// CHECK: vcvtph2uw  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x48,0x7d,0x71,0x7f]
          vcvtph2uw  8128(%rcx), %zmm30

// CHECK: vcvtph2uw  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7c,0xdf,0x7d,0x72,0x80]
          vcvtph2uw  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vcvtph2w %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x48,0x7d,0xf5]
          vcvtph2w %zmm29, %zmm30

// CHECK: vcvtph2w {rn-sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x18,0x7d,0xf5]
          vcvtph2w {rn-sae}, %zmm29, %zmm30

// CHECK: vcvtph2w  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7d,0x4f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtph2w  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtph2w  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7d,0x58,0x7d,0x31]
          vcvtph2w  (%r9){1to32}, %zmm30

// CHECK: vcvtph2w  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7d,0x48,0x7d,0x71,0x7f]
          vcvtph2w  8128(%rcx), %zmm30

// CHECK: vcvtph2w  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7d,0xdf,0x7d,0x72,0x80]
          vcvtph2w  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vcvtps2phx %zmm29, %ymm30
// CHECK: encoding: [0x62,0x05,0x7d,0x48,0x1d,0xf5]
          vcvtps2phx %zmm29, %ymm30

// CHECK: vcvtps2phx {rn-sae}, %zmm29, %ymm30
// CHECK: encoding: [0x62,0x05,0x7d,0x18,0x1d,0xf5]
          vcvtps2phx {rn-sae}, %zmm29, %ymm30

// CHECK: vcvtps2phx  268435456(%rbp,%r14,8), %ymm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7d,0x4f,0x1d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtps2phx  268435456(%rbp,%r14,8), %ymm30 {%k7}

// CHECK: vcvtps2phx  (%r9){1to16}, %ymm30
// CHECK: encoding: [0x62,0x45,0x7d,0x58,0x1d,0x31]
          vcvtps2phx  (%r9){1to16}, %ymm30

// CHECK: vcvtps2phx  8128(%rcx), %ymm30
// CHECK: encoding: [0x62,0x65,0x7d,0x48,0x1d,0x71,0x7f]
          vcvtps2phx  8128(%rcx), %ymm30

// CHECK: vcvtps2phx  -512(%rdx){1to16}, %ymm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7d,0xdf,0x1d,0x72,0x80]
          vcvtps2phx  -512(%rdx){1to16}, %ymm30 {%k7} {z}

// CHECK: vcvtqq2ph %zmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0xfc,0x48,0x5b,0xf5]
          vcvtqq2ph %zmm29, %xmm30

// CHECK: vcvtqq2ph {rn-sae}, %zmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0xfc,0x18,0x5b,0xf5]
          vcvtqq2ph {rn-sae}, %zmm29, %xmm30

// CHECK: vcvtqq2phz  268435456(%rbp,%r14,8), %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0xfc,0x4f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtqq2phz  268435456(%rbp,%r14,8), %xmm30 {%k7}

// CHECK: vcvtqq2ph  (%r9){1to8}, %xmm30
// CHECK: encoding: [0x62,0x45,0xfc,0x58,0x5b,0x31]
          vcvtqq2ph  (%r9){1to8}, %xmm30

// CHECK: vcvtqq2phz  8128(%rcx), %xmm30
// CHECK: encoding: [0x62,0x65,0xfc,0x48,0x5b,0x71,0x7f]
          vcvtqq2phz  8128(%rcx), %xmm30

// CHECK: vcvtqq2ph  -1024(%rdx){1to8}, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0xfc,0xdf,0x5b,0x72,0x80]
          vcvtqq2ph  -1024(%rdx){1to8}, %xmm30 {%k7} {z}

// CHECK: vcvtsd2sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x97,0x00,0x5a,0xf4]
          vcvtsd2sh %xmm28, %xmm29, %xmm30

// CHECK: vcvtsd2sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x97,0x10,0x5a,0xf4]
          vcvtsd2sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vcvtsd2sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x97,0x07,0x5a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtsd2sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vcvtsd2sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x97,0x00,0x5a,0x31]
          vcvtsd2sh  (%r9), %xmm29, %xmm30

// CHECK: vcvtsd2sh  1016(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x97,0x00,0x5a,0x71,0x7f]
          vcvtsd2sh  1016(%rcx), %xmm29, %xmm30

// CHECK: vcvtsd2sh  -1024(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x97,0x87,0x5a,0x72,0x80]
          vcvtsd2sh  -1024(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vcvtsh2sd %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x00,0x5a,0xf4]
          vcvtsh2sd %xmm28, %xmm29, %xmm30

// CHECK: vcvtsh2sd {sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x10,0x5a,0xf4]
          vcvtsh2sd {sae}, %xmm28, %xmm29, %xmm30

// CHECK: vcvtsh2sd  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x16,0x07,0x5a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtsh2sd  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vcvtsh2sd  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x5a,0x31]
          vcvtsh2sd  (%r9), %xmm29, %xmm30

// CHECK: vcvtsh2sd  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x5a,0x71,0x7f]
          vcvtsh2sd  254(%rcx), %xmm29, %xmm30

// CHECK: vcvtsh2sd  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x16,0x87,0x5a,0x72,0x80]
          vcvtsh2sd  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vcvtsh2si %xmm30, %edx
// CHECK: encoding: [0x62,0x95,0x7e,0x08,0x2d,0xd6]
          vcvtsh2si %xmm30, %edx

// CHECK: vcvtsh2si {rn-sae}, %xmm30, %edx
// CHECK: encoding: [0x62,0x95,0x7e,0x18,0x2d,0xd6]
          vcvtsh2si {rn-sae}, %xmm30, %edx

// CHECK: vcvtsh2si %xmm30, %r12
// CHECK: encoding: [0x62,0x15,0xfe,0x08,0x2d,0xe6]
          vcvtsh2si %xmm30, %r12

// CHECK: vcvtsh2si {rn-sae}, %xmm30, %r12
// CHECK: encoding: [0x62,0x15,0xfe,0x18,0x2d,0xe6]
          vcvtsh2si {rn-sae}, %xmm30, %r12

// CHECK: vcvtsh2si  268435456(%rbp,%r14,8), %edx
// CHECK: encoding: [0x62,0xb5,0x7e,0x08,0x2d,0x94,0xf5,0x00,0x00,0x00,0x10]
          vcvtsh2si  268435456(%rbp,%r14,8), %edx

// CHECK: vcvtsh2si  (%r9), %edx
// CHECK: encoding: [0x62,0xd5,0x7e,0x08,0x2d,0x11]
          vcvtsh2si  (%r9), %edx

// CHECK: vcvtsh2si  254(%rcx), %edx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2d,0x51,0x7f]
          vcvtsh2si  254(%rcx), %edx

// CHECK: vcvtsh2si  -256(%rdx), %edx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2d,0x52,0x80]
          vcvtsh2si  -256(%rdx), %edx

// CHECK: vcvtsh2si  268435456(%rbp,%r14,8), %r12
// CHECK: encoding: [0x62,0x35,0xfe,0x08,0x2d,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vcvtsh2si  268435456(%rbp,%r14,8), %r12

// CHECK: vcvtsh2si  (%r9), %r12
// CHECK: encoding: [0x62,0x55,0xfe,0x08,0x2d,0x21]
          vcvtsh2si  (%r9), %r12

// CHECK: vcvtsh2si  254(%rcx), %r12
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x2d,0x61,0x7f]
          vcvtsh2si  254(%rcx), %r12

// CHECK: vcvtsh2si  -256(%rdx), %r12
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x2d,0x62,0x80]
          vcvtsh2si  -256(%rdx), %r12

// CHECK: vcvtsh2ss %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x14,0x00,0x13,0xf4]
          vcvtsh2ss %xmm28, %xmm29, %xmm30

// CHECK: vcvtsh2ss {sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x14,0x10,0x13,0xf4]
          vcvtsh2ss {sae}, %xmm28, %xmm29, %xmm30

// CHECK: vcvtsh2ss  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x14,0x07,0x13,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtsh2ss  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vcvtsh2ss  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x14,0x00,0x13,0x31]
          vcvtsh2ss  (%r9), %xmm29, %xmm30

// CHECK: vcvtsh2ss  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x14,0x00,0x13,0x71,0x7f]
          vcvtsh2ss  254(%rcx), %xmm29, %xmm30

// CHECK: vcvtsh2ss  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x14,0x87,0x13,0x72,0x80]
          vcvtsh2ss  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vcvtsh2usi %xmm30, %edx
// CHECK: encoding: [0x62,0x95,0x7e,0x08,0x79,0xd6]
          vcvtsh2usi %xmm30, %edx

// CHECK: vcvtsh2usi {rn-sae}, %xmm30, %edx
// CHECK: encoding: [0x62,0x95,0x7e,0x18,0x79,0xd6]
          vcvtsh2usi {rn-sae}, %xmm30, %edx

// CHECK: vcvtsh2usi %xmm30, %r12
// CHECK: encoding: [0x62,0x15,0xfe,0x08,0x79,0xe6]
          vcvtsh2usi %xmm30, %r12

// CHECK: vcvtsh2usi {rn-sae}, %xmm30, %r12
// CHECK: encoding: [0x62,0x15,0xfe,0x18,0x79,0xe6]
          vcvtsh2usi {rn-sae}, %xmm30, %r12

// CHECK: vcvtsh2usi  268435456(%rbp,%r14,8), %edx
// CHECK: encoding: [0x62,0xb5,0x7e,0x08,0x79,0x94,0xf5,0x00,0x00,0x00,0x10]
          vcvtsh2usi  268435456(%rbp,%r14,8), %edx

// CHECK: vcvtsh2usi  (%r9), %edx
// CHECK: encoding: [0x62,0xd5,0x7e,0x08,0x79,0x11]
          vcvtsh2usi  (%r9), %edx

// CHECK: vcvtsh2usi  254(%rcx), %edx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x79,0x51,0x7f]
          vcvtsh2usi  254(%rcx), %edx

// CHECK: vcvtsh2usi  -256(%rdx), %edx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x79,0x52,0x80]
          vcvtsh2usi  -256(%rdx), %edx

// CHECK: vcvtsh2usi  268435456(%rbp,%r14,8), %r12
// CHECK: encoding: [0x62,0x35,0xfe,0x08,0x79,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vcvtsh2usi  268435456(%rbp,%r14,8), %r12

// CHECK: vcvtsh2usi  (%r9), %r12
// CHECK: encoding: [0x62,0x55,0xfe,0x08,0x79,0x21]
          vcvtsh2usi  (%r9), %r12

// CHECK: vcvtsh2usi  254(%rcx), %r12
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x79,0x61,0x7f]
          vcvtsh2usi  254(%rcx), %r12

// CHECK: vcvtsh2usi  -256(%rdx), %r12
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x79,0x62,0x80]
          vcvtsh2usi  -256(%rdx), %r12

// CHECK: vcvtsi2sh %r12, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x96,0x00,0x2a,0xf4]
          vcvtsi2sh %r12, %xmm29, %xmm30

// CHECK: vcvtsi2sh %r12, {rn-sae}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x96,0x10,0x2a,0xf4]
          vcvtsi2sh %r12, {rn-sae}, %xmm29, %xmm30

// CHECK: vcvtsi2sh %edx, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x2a,0xf2]
          vcvtsi2sh %edx, %xmm29, %xmm30

// CHECK: vcvtsi2sh %edx, {rn-sae}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x10,0x2a,0xf2]
          vcvtsi2sh %edx, {rn-sae}, %xmm29, %xmm30

// CHECK: vcvtsi2shl  268435456(%rbp,%r14,8), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x25,0x16,0x00,0x2a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtsi2shl  268435456(%rbp,%r14,8), %xmm29, %xmm30

// CHECK: vcvtsi2shl  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x2a,0x31]
          vcvtsi2shl  (%r9), %xmm29, %xmm30

// CHECK: vcvtsi2shl  508(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x2a,0x71,0x7f]
          vcvtsi2shl  508(%rcx), %xmm29, %xmm30

// CHECK: vcvtsi2shl  -512(%rdx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x2a,0x72,0x80]
          vcvtsi2shl  -512(%rdx), %xmm29, %xmm30

// CHECK: vcvtsi2shq  1016(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x96,0x00,0x2a,0x71,0x7f]
          vcvtsi2shq  1016(%rcx), %xmm29, %xmm30

// CHECK: vcvtsi2shq  -1024(%rdx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x96,0x00,0x2a,0x72,0x80]
          vcvtsi2shq  -1024(%rdx), %xmm29, %xmm30

// CHECK: vcvtss2sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x14,0x00,0x1d,0xf4]
          vcvtss2sh %xmm28, %xmm29, %xmm30

// CHECK: vcvtss2sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x14,0x10,0x1d,0xf4]
          vcvtss2sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vcvtss2sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x14,0x07,0x1d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtss2sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vcvtss2sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x14,0x00,0x1d,0x31]
          vcvtss2sh  (%r9), %xmm29, %xmm30

// CHECK: vcvtss2sh  508(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x14,0x00,0x1d,0x71,0x7f]
          vcvtss2sh  508(%rcx), %xmm29, %xmm30

// CHECK: vcvtss2sh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x14,0x87,0x1d,0x72,0x80]
          vcvtss2sh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vcvttph2dq %ymm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7e,0x48,0x5b,0xf5]
          vcvttph2dq %ymm29, %zmm30

// CHECK: vcvttph2dq {sae}, %ymm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7e,0x18,0x5b,0xf5]
          vcvttph2dq {sae}, %ymm29, %zmm30

// CHECK: vcvttph2dq  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7e,0x4f,0x5b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2dq  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvttph2dq  (%r9){1to16}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7e,0x58,0x5b,0x31]
          vcvttph2dq  (%r9){1to16}, %zmm30

// CHECK: vcvttph2dq  4064(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7e,0x48,0x5b,0x71,0x7f]
          vcvttph2dq  4064(%rcx), %zmm30

// CHECK: vcvttph2dq  -256(%rdx){1to16}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7e,0xdf,0x5b,0x72,0x80]
          vcvttph2dq  -256(%rdx){1to16}, %zmm30 {%k7} {z}

// CHECK: vcvttph2qq %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x48,0x7a,0xf5]
          vcvttph2qq %xmm29, %zmm30

// CHECK: vcvttph2qq {sae}, %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x18,0x7a,0xf5]
          vcvttph2qq {sae}, %xmm29, %zmm30

// CHECK: vcvttph2qq  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7d,0x4f,0x7a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2qq  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvttph2qq  (%r9){1to8}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7d,0x58,0x7a,0x31]
          vcvttph2qq  (%r9){1to8}, %zmm30

// CHECK: vcvttph2qq  2032(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7d,0x48,0x7a,0x71,0x7f]
          vcvttph2qq  2032(%rcx), %zmm30

// CHECK: vcvttph2qq  -256(%rdx){1to8}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7d,0xdf,0x7a,0x72,0x80]
          vcvttph2qq  -256(%rdx){1to8}, %zmm30 {%k7} {z}

// CHECK: vcvttph2udq %ymm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x48,0x78,0xf5]
          vcvttph2udq %ymm29, %zmm30

// CHECK: vcvttph2udq {sae}, %ymm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x18,0x78,0xf5]
          vcvttph2udq {sae}, %ymm29, %zmm30

// CHECK: vcvttph2udq  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7c,0x4f,0x78,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2udq  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvttph2udq  (%r9){1to16}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7c,0x58,0x78,0x31]
          vcvttph2udq  (%r9){1to16}, %zmm30

// CHECK: vcvttph2udq  4064(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x48,0x78,0x71,0x7f]
          vcvttph2udq  4064(%rcx), %zmm30

// CHECK: vcvttph2udq  -256(%rdx){1to16}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7c,0xdf,0x78,0x72,0x80]
          vcvttph2udq  -256(%rdx){1to16}, %zmm30 {%k7} {z}

// CHECK: vcvttph2uqq %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x48,0x78,0xf5]
          vcvttph2uqq %xmm29, %zmm30

// CHECK: vcvttph2uqq {sae}, %xmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x18,0x78,0xf5]
          vcvttph2uqq {sae}, %xmm29, %zmm30

// CHECK: vcvttph2uqq  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7d,0x4f,0x78,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2uqq  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvttph2uqq  (%r9){1to8}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7d,0x58,0x78,0x31]
          vcvttph2uqq  (%r9){1to8}, %zmm30

// CHECK: vcvttph2uqq  2032(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7d,0x48,0x78,0x71,0x7f]
          vcvttph2uqq  2032(%rcx), %zmm30

// CHECK: vcvttph2uqq  -256(%rdx){1to8}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7d,0xdf,0x78,0x72,0x80]
          vcvttph2uqq  -256(%rdx){1to8}, %zmm30 {%k7} {z}

// CHECK: vcvttph2uw %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x48,0x7c,0xf5]
          vcvttph2uw %zmm29, %zmm30

// CHECK: vcvttph2uw {sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x18,0x7c,0xf5]
          vcvttph2uw {sae}, %zmm29, %zmm30

// CHECK: vcvttph2uw  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7c,0x4f,0x7c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2uw  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvttph2uw  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7c,0x58,0x7c,0x31]
          vcvttph2uw  (%r9){1to32}, %zmm30

// CHECK: vcvttph2uw  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x48,0x7c,0x71,0x7f]
          vcvttph2uw  8128(%rcx), %zmm30

// CHECK: vcvttph2uw  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7c,0xdf,0x7c,0x72,0x80]
          vcvttph2uw  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vcvttph2w %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x48,0x7c,0xf5]
          vcvttph2w %zmm29, %zmm30

// CHECK: vcvttph2w {sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7d,0x18,0x7c,0xf5]
          vcvttph2w {sae}, %zmm29, %zmm30

// CHECK: vcvttph2w  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7d,0x4f,0x7c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvttph2w  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvttph2w  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7d,0x58,0x7c,0x31]
          vcvttph2w  (%r9){1to32}, %zmm30

// CHECK: vcvttph2w  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7d,0x48,0x7c,0x71,0x7f]
          vcvttph2w  8128(%rcx), %zmm30

// CHECK: vcvttph2w  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7d,0xdf,0x7c,0x72,0x80]
          vcvttph2w  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vcvttsh2si %xmm30, %edx
// CHECK: encoding: [0x62,0x95,0x7e,0x08,0x2c,0xd6]
          vcvttsh2si %xmm30, %edx

// CHECK: vcvttsh2si {sae}, %xmm30, %edx
// CHECK: encoding: [0x62,0x95,0x7e,0x18,0x2c,0xd6]
          vcvttsh2si {sae}, %xmm30, %edx

// CHECK: vcvttsh2si %xmm30, %r12
// CHECK: encoding: [0x62,0x15,0xfe,0x08,0x2c,0xe6]
          vcvttsh2si %xmm30, %r12

// CHECK: vcvttsh2si {sae}, %xmm30, %r12
// CHECK: encoding: [0x62,0x15,0xfe,0x18,0x2c,0xe6]
          vcvttsh2si {sae}, %xmm30, %r12

// CHECK: vcvttsh2si  268435456(%rbp,%r14,8), %edx
// CHECK: encoding: [0x62,0xb5,0x7e,0x08,0x2c,0x94,0xf5,0x00,0x00,0x00,0x10]
          vcvttsh2si  268435456(%rbp,%r14,8), %edx

// CHECK: vcvttsh2si  (%r9), %edx
// CHECK: encoding: [0x62,0xd5,0x7e,0x08,0x2c,0x11]
          vcvttsh2si  (%r9), %edx

// CHECK: vcvttsh2si  254(%rcx), %edx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2c,0x51,0x7f]
          vcvttsh2si  254(%rcx), %edx

// CHECK: vcvttsh2si  -256(%rdx), %edx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x2c,0x52,0x80]
          vcvttsh2si  -256(%rdx), %edx

// CHECK: vcvttsh2si  268435456(%rbp,%r14,8), %r12
// CHECK: encoding: [0x62,0x35,0xfe,0x08,0x2c,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vcvttsh2si  268435456(%rbp,%r14,8), %r12

// CHECK: vcvttsh2si  (%r9), %r12
// CHECK: encoding: [0x62,0x55,0xfe,0x08,0x2c,0x21]
          vcvttsh2si  (%r9), %r12

// CHECK: vcvttsh2si  254(%rcx), %r12
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x2c,0x61,0x7f]
          vcvttsh2si  254(%rcx), %r12

// CHECK: vcvttsh2si  -256(%rdx), %r12
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x2c,0x62,0x80]
          vcvttsh2si  -256(%rdx), %r12

// CHECK: vcvttsh2usi %xmm30, %edx
// CHECK: encoding: [0x62,0x95,0x7e,0x08,0x78,0xd6]
          vcvttsh2usi %xmm30, %edx

// CHECK: vcvttsh2usi {sae}, %xmm30, %edx
// CHECK: encoding: [0x62,0x95,0x7e,0x18,0x78,0xd6]
          vcvttsh2usi {sae}, %xmm30, %edx

// CHECK: vcvttsh2usi %xmm30, %r12
// CHECK: encoding: [0x62,0x15,0xfe,0x08,0x78,0xe6]
          vcvttsh2usi %xmm30, %r12

// CHECK: vcvttsh2usi {sae}, %xmm30, %r12
// CHECK: encoding: [0x62,0x15,0xfe,0x18,0x78,0xe6]
          vcvttsh2usi {sae}, %xmm30, %r12

// CHECK: vcvttsh2usi  268435456(%rbp,%r14,8), %edx
// CHECK: encoding: [0x62,0xb5,0x7e,0x08,0x78,0x94,0xf5,0x00,0x00,0x00,0x10]
          vcvttsh2usi  268435456(%rbp,%r14,8), %edx

// CHECK: vcvttsh2usi  (%r9), %edx
// CHECK: encoding: [0x62,0xd5,0x7e,0x08,0x78,0x11]
          vcvttsh2usi  (%r9), %edx

// CHECK: vcvttsh2usi  254(%rcx), %edx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x78,0x51,0x7f]
          vcvttsh2usi  254(%rcx), %edx

// CHECK: vcvttsh2usi  -256(%rdx), %edx
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x78,0x52,0x80]
          vcvttsh2usi  -256(%rdx), %edx

// CHECK: vcvttsh2usi  268435456(%rbp,%r14,8), %r12
// CHECK: encoding: [0x62,0x35,0xfe,0x08,0x78,0xa4,0xf5,0x00,0x00,0x00,0x10]
          vcvttsh2usi  268435456(%rbp,%r14,8), %r12

// CHECK: vcvttsh2usi  (%r9), %r12
// CHECK: encoding: [0x62,0x55,0xfe,0x08,0x78,0x21]
          vcvttsh2usi  (%r9), %r12

// CHECK: vcvttsh2usi  254(%rcx), %r12
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x78,0x61,0x7f]
          vcvttsh2usi  254(%rcx), %r12

// CHECK: vcvttsh2usi  -256(%rdx), %r12
// CHECK: encoding: [0x62,0x75,0xfe,0x08,0x78,0x62,0x80]
          vcvttsh2usi  -256(%rdx), %r12

// CHECK: vcvtudq2ph %zmm29, %ymm30
// CHECK: encoding: [0x62,0x05,0x7f,0x48,0x7a,0xf5]
          vcvtudq2ph %zmm29, %ymm30

// CHECK: vcvtudq2ph {rn-sae}, %zmm29, %ymm30
// CHECK: encoding: [0x62,0x05,0x7f,0x18,0x7a,0xf5]
          vcvtudq2ph {rn-sae}, %zmm29, %ymm30

// CHECK: vcvtudq2ph  268435456(%rbp,%r14,8), %ymm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7f,0x4f,0x7a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtudq2ph  268435456(%rbp,%r14,8), %ymm30 {%k7}

// CHECK: vcvtudq2ph  (%r9){1to16}, %ymm30
// CHECK: encoding: [0x62,0x45,0x7f,0x58,0x7a,0x31]
          vcvtudq2ph  (%r9){1to16}, %ymm30

// CHECK: vcvtudq2ph  8128(%rcx), %ymm30
// CHECK: encoding: [0x62,0x65,0x7f,0x48,0x7a,0x71,0x7f]
          vcvtudq2ph  8128(%rcx), %ymm30

// CHECK: vcvtudq2ph  -512(%rdx){1to16}, %ymm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7f,0xdf,0x7a,0x72,0x80]
          vcvtudq2ph  -512(%rdx){1to16}, %ymm30 {%k7} {z}

// CHECK: vcvtuqq2ph %zmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0xff,0x48,0x7a,0xf5]
          vcvtuqq2ph %zmm29, %xmm30

// CHECK: vcvtuqq2ph {rn-sae}, %zmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0xff,0x18,0x7a,0xf5]
          vcvtuqq2ph {rn-sae}, %zmm29, %xmm30

// CHECK: vcvtuqq2phz  268435456(%rbp,%r14,8), %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0xff,0x4f,0x7a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtuqq2phz  268435456(%rbp,%r14,8), %xmm30 {%k7}

// CHECK: vcvtuqq2ph  (%r9){1to8}, %xmm30
// CHECK: encoding: [0x62,0x45,0xff,0x58,0x7a,0x31]
          vcvtuqq2ph  (%r9){1to8}, %xmm30

// CHECK: vcvtuqq2phz  8128(%rcx), %xmm30
// CHECK: encoding: [0x62,0x65,0xff,0x48,0x7a,0x71,0x7f]
          vcvtuqq2phz  8128(%rcx), %xmm30

// CHECK: vcvtuqq2ph  -1024(%rdx){1to8}, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0xff,0xdf,0x7a,0x72,0x80]
          vcvtuqq2ph  -1024(%rdx){1to8}, %xmm30 {%k7} {z}

// CHECK: vcvtusi2sh %r12, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x96,0x00,0x7b,0xf4]
          vcvtusi2sh %r12, %xmm29, %xmm30

// CHECK: vcvtusi2sh %r12, {rn-sae}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x96,0x10,0x7b,0xf4]
          vcvtusi2sh %r12, {rn-sae}, %xmm29, %xmm30

// CHECK: vcvtusi2sh %edx, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x7b,0xf2]
          vcvtusi2sh %edx, %xmm29, %xmm30

// CHECK: vcvtusi2sh %edx, {rn-sae}, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x10,0x7b,0xf2]
          vcvtusi2sh %edx, {rn-sae}, %xmm29, %xmm30

// CHECK: vcvtusi2shl  268435456(%rbp,%r14,8), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x25,0x16,0x00,0x7b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtusi2shl  268435456(%rbp,%r14,8), %xmm29, %xmm30

// CHECK: vcvtusi2shl  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x7b,0x31]
          vcvtusi2shl  (%r9), %xmm29, %xmm30

// CHECK: vcvtusi2shl  508(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x7b,0x71,0x7f]
          vcvtusi2shl  508(%rcx), %xmm29, %xmm30

// CHECK: vcvtusi2shl  -512(%rdx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x7b,0x72,0x80]
          vcvtusi2shl  -512(%rdx), %xmm29, %xmm30

// CHECK: vcvtusi2shq  1016(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x96,0x00,0x7b,0x71,0x7f]
          vcvtusi2shq  1016(%rcx), %xmm29, %xmm30

// CHECK: vcvtusi2shq  -1024(%rdx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x96,0x00,0x7b,0x72,0x80]
          vcvtusi2shq  -1024(%rdx), %xmm29, %xmm30

// CHECK: vcvtuw2ph %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7f,0x48,0x7d,0xf5]
          vcvtuw2ph %zmm29, %zmm30

// CHECK: vcvtuw2ph {rn-sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7f,0x18,0x7d,0xf5]
          vcvtuw2ph {rn-sae}, %zmm29, %zmm30

// CHECK: vcvtuw2ph  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7f,0x4f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtuw2ph  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtuw2ph  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7f,0x58,0x7d,0x31]
          vcvtuw2ph  (%r9){1to32}, %zmm30

// CHECK: vcvtuw2ph  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7f,0x48,0x7d,0x71,0x7f]
          vcvtuw2ph  8128(%rcx), %zmm30

// CHECK: vcvtuw2ph  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7f,0xdf,0x7d,0x72,0x80]
          vcvtuw2ph  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vcvtw2ph %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7e,0x48,0x7d,0xf5]
          vcvtw2ph %zmm29, %zmm30

// CHECK: vcvtw2ph {rn-sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7e,0x18,0x7d,0xf5]
          vcvtw2ph {rn-sae}, %zmm29, %zmm30

// CHECK: vcvtw2ph  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7e,0x4f,0x7d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vcvtw2ph  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vcvtw2ph  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7e,0x58,0x7d,0x31]
          vcvtw2ph  (%r9){1to32}, %zmm30

// CHECK: vcvtw2ph  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7e,0x48,0x7d,0x71,0x7f]
          vcvtw2ph  8128(%rcx), %zmm30

// CHECK: vcvtw2ph  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7e,0xdf,0x7d,0x72,0x80]
          vcvtw2ph  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vfpclassph $123, %zmm30, %k5
// CHECK: encoding: [0x62,0x93,0x7c,0x48,0x66,0xee,0x7b]
          vfpclassph $123, %zmm30, %k5

// CHECK: vfpclassphz  $123, 268435456(%rbp,%r14,8), %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x7c,0x4f,0x66,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vfpclassphz  $123, 268435456(%rbp,%r14,8), %k5 {%k7}

// CHECK: vfpclassph  $123, (%r9){1to32}, %k5
// CHECK: encoding: [0x62,0xd3,0x7c,0x58,0x66,0x29,0x7b]
          vfpclassph  $123, (%r9){1to32}, %k5

// CHECK: vfpclassphz  $123, 8128(%rcx), %k5
// CHECK: encoding: [0x62,0xf3,0x7c,0x48,0x66,0x69,0x7f,0x7b]
          vfpclassphz  $123, 8128(%rcx), %k5

// CHECK: vfpclassph  $123, -256(%rdx){1to32}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x5f,0x66,0x6a,0x80,0x7b]
          vfpclassph  $123, -256(%rdx){1to32}, %k5 {%k7}

// CHECK: vfpclasssh $123, %xmm30, %k5
// CHECK: encoding: [0x62,0x93,0x7c,0x08,0x67,0xee,0x7b]
          vfpclasssh $123, %xmm30, %k5

// CHECK: vfpclasssh  $123, 268435456(%rbp,%r14,8), %k5 {%k7}
// CHECK: encoding: [0x62,0xb3,0x7c,0x0f,0x67,0xac,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vfpclasssh  $123, 268435456(%rbp,%r14,8), %k5 {%k7}

// CHECK: vfpclasssh  $123, (%r9), %k5
// CHECK: encoding: [0x62,0xd3,0x7c,0x08,0x67,0x29,0x7b]
          vfpclasssh  $123, (%r9), %k5

// CHECK: vfpclasssh  $123, 254(%rcx), %k5
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x67,0x69,0x7f,0x7b]
          vfpclasssh  $123, 254(%rcx), %k5

// CHECK: vfpclasssh  $123, -256(%rdx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x0f,0x67,0x6a,0x80,0x7b]
          vfpclasssh  $123, -256(%rdx), %k5 {%k7}

// CHECK: vgetexpph %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x7d,0x48,0x42,0xf5]
          vgetexpph %zmm29, %zmm30

// CHECK: vgetexpph {sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x7d,0x18,0x42,0xf5]
          vgetexpph {sae}, %zmm29, %zmm30

// CHECK: vgetexpph  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x7d,0x4f,0x42,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpph  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vgetexpph  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x46,0x7d,0x58,0x42,0x31]
          vgetexpph  (%r9){1to32}, %zmm30

// CHECK: vgetexpph  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x66,0x7d,0x48,0x42,0x71,0x7f]
          vgetexpph  8128(%rcx), %zmm30

// CHECK: vgetexpph  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x7d,0xdf,0x42,0x72,0x80]
          vgetexpph  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vgetexpsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x43,0xf4]
          vgetexpsh %xmm28, %xmm29, %xmm30

// CHECK: vgetexpsh {sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x43,0xf4]
          vgetexpsh {sae}, %xmm28, %xmm29, %xmm30

// CHECK: vgetexpsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x43,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vgetexpsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vgetexpsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0x43,0x31]
          vgetexpsh  (%r9), %xmm29, %xmm30

// CHECK: vgetexpsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x43,0x71,0x7f]
          vgetexpsh  254(%rcx), %xmm29, %xmm30

// CHECK: vgetexpsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0x43,0x72,0x80]
          vgetexpsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vgetmantph $123, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x03,0x7c,0x48,0x26,0xf5,0x7b]
          vgetmantph $123, %zmm29, %zmm30

// CHECK: vgetmantph $123, {sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x03,0x7c,0x18,0x26,0xf5,0x7b]
          vgetmantph $123, {sae}, %zmm29, %zmm30

// CHECK: vgetmantph  $123, 268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x23,0x7c,0x4f,0x26,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantph  $123, 268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vgetmantph  $123, (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x43,0x7c,0x58,0x26,0x31,0x7b]
          vgetmantph  $123, (%r9){1to32}, %zmm30

// CHECK: vgetmantph  $123, 8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x63,0x7c,0x48,0x26,0x71,0x7f,0x7b]
          vgetmantph  $123, 8128(%rcx), %zmm30

// CHECK: vgetmantph  $123, -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x63,0x7c,0xdf,0x26,0x72,0x80,0x7b]
          vgetmantph  $123, -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vgetmantsh $123, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x03,0x14,0x00,0x27,0xf4,0x7b]
          vgetmantsh $123, %xmm28, %xmm29, %xmm30

// CHECK: vgetmantsh $123, {sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x03,0x14,0x10,0x27,0xf4,0x7b]
          vgetmantsh $123, {sae}, %xmm28, %xmm29, %xmm30

// CHECK: vgetmantsh  $123, 268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x23,0x14,0x07,0x27,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vgetmantsh  $123, 268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vgetmantsh  $123, (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x43,0x14,0x00,0x27,0x31,0x7b]
          vgetmantsh  $123, (%r9), %xmm29, %xmm30

// CHECK: vgetmantsh  $123, 254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x63,0x14,0x00,0x27,0x71,0x7f,0x7b]
          vgetmantsh  $123, 254(%rcx), %xmm29, %xmm30

// CHECK: vgetmantsh  $123, -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x63,0x14,0x87,0x27,0x72,0x80,0x7b]
          vgetmantsh  $123, -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vrcpph %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x7d,0x48,0x4c,0xf5]
          vrcpph %zmm29, %zmm30

// CHECK: vrcpph  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x7d,0x4f,0x4c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpph  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vrcpph  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x46,0x7d,0x58,0x4c,0x31]
          vrcpph  (%r9){1to32}, %zmm30

// CHECK: vrcpph  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x66,0x7d,0x48,0x4c,0x71,0x7f]
          vrcpph  8128(%rcx), %zmm30

// CHECK: vrcpph  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x7d,0xdf,0x4c,0x72,0x80]
          vrcpph  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vrcpsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x4d,0xf4]
          vrcpsh %xmm28, %xmm29, %xmm30

// CHECK: vrcpsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x4d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrcpsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vrcpsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0x4d,0x31]
          vrcpsh  (%r9), %xmm29, %xmm30

// CHECK: vrcpsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x4d,0x71,0x7f]
          vrcpsh  254(%rcx), %xmm29, %xmm30

// CHECK: vrcpsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0x4d,0x72,0x80]
          vrcpsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vreduceph $123, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x03,0x7c,0x48,0x56,0xf5,0x7b]
          vreduceph $123, %zmm29, %zmm30

// CHECK: vreduceph $123, {sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x03,0x7c,0x18,0x56,0xf5,0x7b]
          vreduceph $123, {sae}, %zmm29, %zmm30

// CHECK: vreduceph  $123, 268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x23,0x7c,0x4f,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreduceph  $123, 268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vreduceph  $123, (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x43,0x7c,0x58,0x56,0x31,0x7b]
          vreduceph  $123, (%r9){1to32}, %zmm30

// CHECK: vreduceph  $123, 8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x63,0x7c,0x48,0x56,0x71,0x7f,0x7b]
          vreduceph  $123, 8128(%rcx), %zmm30

// CHECK: vreduceph  $123, -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x63,0x7c,0xdf,0x56,0x72,0x80,0x7b]
          vreduceph  $123, -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vreducesh $123, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x03,0x14,0x00,0x57,0xf4,0x7b]
          vreducesh $123, %xmm28, %xmm29, %xmm30

// CHECK: vreducesh $123, {sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x03,0x14,0x10,0x57,0xf4,0x7b]
          vreducesh $123, {sae}, %xmm28, %xmm29, %xmm30

// CHECK: vreducesh  $123, 268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x23,0x14,0x07,0x57,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vreducesh  $123, 268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vreducesh  $123, (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x43,0x14,0x00,0x57,0x31,0x7b]
          vreducesh  $123, (%r9), %xmm29, %xmm30

// CHECK: vreducesh  $123, 254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x63,0x14,0x00,0x57,0x71,0x7f,0x7b]
          vreducesh  $123, 254(%rcx), %xmm29, %xmm30

// CHECK: vreducesh  $123, -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x63,0x14,0x87,0x57,0x72,0x80,0x7b]
          vreducesh  $123, -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vrndscaleph $123, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x03,0x7c,0x48,0x08,0xf5,0x7b]
          vrndscaleph $123, %zmm29, %zmm30

// CHECK: vrndscaleph $123, {sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x03,0x7c,0x18,0x08,0xf5,0x7b]
          vrndscaleph $123, {sae}, %zmm29, %zmm30

// CHECK: vrndscaleph  $123, 268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x23,0x7c,0x4f,0x08,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscaleph  $123, 268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vrndscaleph  $123, (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x43,0x7c,0x58,0x08,0x31,0x7b]
          vrndscaleph  $123, (%r9){1to32}, %zmm30

// CHECK: vrndscaleph  $123, 8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x63,0x7c,0x48,0x08,0x71,0x7f,0x7b]
          vrndscaleph  $123, 8128(%rcx), %zmm30

// CHECK: vrndscaleph  $123, -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x63,0x7c,0xdf,0x08,0x72,0x80,0x7b]
          vrndscaleph  $123, -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vrndscalesh $123, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x03,0x14,0x00,0x0a,0xf4,0x7b]
          vrndscalesh $123, %xmm28, %xmm29, %xmm30

// CHECK: vrndscalesh $123, {sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x03,0x14,0x10,0x0a,0xf4,0x7b]
          vrndscalesh $123, {sae}, %xmm28, %xmm29, %xmm30

// CHECK: vrndscalesh  $123, 268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x23,0x14,0x07,0x0a,0xb4,0xf5,0x00,0x00,0x00,0x10,0x7b]
          vrndscalesh  $123, 268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vrndscalesh  $123, (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x43,0x14,0x00,0x0a,0x31,0x7b]
          vrndscalesh  $123, (%r9), %xmm29, %xmm30

// CHECK: vrndscalesh  $123, 254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x63,0x14,0x00,0x0a,0x71,0x7f,0x7b]
          vrndscalesh  $123, 254(%rcx), %xmm29, %xmm30

// CHECK: vrndscalesh  $123, -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x63,0x14,0x87,0x0a,0x72,0x80,0x7b]
          vrndscalesh  $123, -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vrsqrtph %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x7d,0x48,0x4e,0xf5]
          vrsqrtph %zmm29, %zmm30

// CHECK: vrsqrtph  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x7d,0x4f,0x4e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtph  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vrsqrtph  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x46,0x7d,0x58,0x4e,0x31]
          vrsqrtph  (%r9){1to32}, %zmm30

// CHECK: vrsqrtph  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x66,0x7d,0x48,0x4e,0x71,0x7f]
          vrsqrtph  8128(%rcx), %zmm30

// CHECK: vrsqrtph  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x7d,0xdf,0x4e,0x72,0x80]
          vrsqrtph  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vrsqrtsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x4f,0xf4]
          vrsqrtsh %xmm28, %xmm29, %xmm30

// CHECK: vrsqrtsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x4f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vrsqrtsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vrsqrtsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0x4f,0x31]
          vrsqrtsh  (%r9), %xmm29, %xmm30

// CHECK: vrsqrtsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x4f,0x71,0x7f]
          vrsqrtsh  254(%rcx), %xmm29, %xmm30

// CHECK: vrsqrtsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0x4f,0x72,0x80]
          vrsqrtsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vscalefph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0x2c,0xf4]
          vscalefph %zmm28, %zmm29, %zmm30

// CHECK: vscalefph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x2c,0xf4]
          vscalefph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vscalefph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0x2c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vscalefph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0x2c,0x31]
          vscalefph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vscalefph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0x2c,0x71,0x7f]
          vscalefph  8128(%rcx), %zmm29, %zmm30

// CHECK: vscalefph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0x2c,0x72,0x80]
          vscalefph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vscalefsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x2d,0xf4]
          vscalefsh %xmm28, %xmm29, %xmm30

// CHECK: vscalefsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x2d,0xf4]
          vscalefsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vscalefsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x2d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vscalefsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vscalefsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0x2d,0x31]
          vscalefsh  (%r9), %xmm29, %xmm30

// CHECK: vscalefsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x2d,0x71,0x7f]
          vscalefsh  254(%rcx), %xmm29, %xmm30

// CHECK: vscalefsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0x2d,0x72,0x80]
          vscalefsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vsqrtph %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x48,0x51,0xf5]
          vsqrtph %zmm29, %zmm30

// CHECK: vsqrtph {rn-sae}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x05,0x7c,0x18,0x51,0xf5]
          vsqrtph {rn-sae}, %zmm29, %zmm30

// CHECK: vsqrtph  268435456(%rbp,%r14,8), %zmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x7c,0x4f,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtph  268435456(%rbp,%r14,8), %zmm30 {%k7}

// CHECK: vsqrtph  (%r9){1to32}, %zmm30
// CHECK: encoding: [0x62,0x45,0x7c,0x58,0x51,0x31]
          vsqrtph  (%r9){1to32}, %zmm30

// CHECK: vsqrtph  8128(%rcx), %zmm30
// CHECK: encoding: [0x62,0x65,0x7c,0x48,0x51,0x71,0x7f]
          vsqrtph  8128(%rcx), %zmm30

// CHECK: vsqrtph  -256(%rdx){1to32}, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x7c,0xdf,0x51,0x72,0x80]
          vsqrtph  -256(%rdx){1to32}, %zmm30 {%k7} {z}

// CHECK: vsqrtsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x00,0x51,0xf4]
          vsqrtsh %xmm28, %xmm29, %xmm30

// CHECK: vsqrtsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x05,0x16,0x10,0x51,0xf4]
          vsqrtsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vsqrtsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x25,0x16,0x07,0x51,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vsqrtsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vsqrtsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x45,0x16,0x00,0x51,0x31]
          vsqrtsh  (%r9), %xmm29, %xmm30

// CHECK: vsqrtsh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x65,0x16,0x00,0x51,0x71,0x7f]
          vsqrtsh  254(%rcx), %xmm29, %xmm30

// CHECK: vsqrtsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x65,0x16,0x87,0x51,0x72,0x80]
          vsqrtsh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfmadd132ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0x98,0xf4]
          vfmadd132ph %zmm28, %zmm29, %zmm30

// CHECK: vfmadd132ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x98,0xf4]
          vfmadd132ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmadd132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0x98,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmadd132ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0x98,0x31]
          vfmadd132ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmadd132ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0x98,0x71,0x7f]
          vfmadd132ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmadd132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0x98,0x72,0x80]
          vfmadd132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmadd132sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x99,0xf4]
          vfmadd132sh %xmm28, %xmm29, %xmm30

// CHECK: vfmadd132sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x99,0xf4]
          vfmadd132sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfmadd132sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x99,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd132sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfmadd132sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0x99,0x31]
          vfmadd132sh  (%r9), %xmm29, %xmm30

// CHECK: vfmadd132sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x99,0x71,0x7f]
          vfmadd132sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfmadd132sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0x99,0x72,0x80]
          vfmadd132sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfmadd213ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xa8,0xf4]
          vfmadd213ph %zmm28, %zmm29, %zmm30

// CHECK: vfmadd213ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xa8,0xf4]
          vfmadd213ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmadd213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xa8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmadd213ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xa8,0x31]
          vfmadd213ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmadd213ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xa8,0x71,0x7f]
          vfmadd213ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmadd213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xa8,0x72,0x80]
          vfmadd213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmadd213sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xa9,0xf4]
          vfmadd213sh %xmm28, %xmm29, %xmm30

// CHECK: vfmadd213sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xa9,0xf4]
          vfmadd213sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfmadd213sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xa9,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd213sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfmadd213sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0xa9,0x31]
          vfmadd213sh  (%r9), %xmm29, %xmm30

// CHECK: vfmadd213sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xa9,0x71,0x7f]
          vfmadd213sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfmadd213sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0xa9,0x72,0x80]
          vfmadd213sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfmadd231ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xb8,0xf4]
          vfmadd231ph %zmm28, %zmm29, %zmm30

// CHECK: vfmadd231ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xb8,0xf4]
          vfmadd231ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmadd231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xb8,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmadd231ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xb8,0x31]
          vfmadd231ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmadd231ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xb8,0x71,0x7f]
          vfmadd231ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmadd231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xb8,0x72,0x80]
          vfmadd231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmadd231sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xb9,0xf4]
          vfmadd231sh %xmm28, %xmm29, %xmm30

// CHECK: vfmadd231sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xb9,0xf4]
          vfmadd231sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfmadd231sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xb9,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmadd231sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfmadd231sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0xb9,0x31]
          vfmadd231sh  (%r9), %xmm29, %xmm30

// CHECK: vfmadd231sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xb9,0x71,0x7f]
          vfmadd231sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfmadd231sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0xb9,0x72,0x80]
          vfmadd231sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfmaddsub132ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0x96,0xf4]
          vfmaddsub132ph %zmm28, %zmm29, %zmm30

// CHECK: vfmaddsub132ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x96,0xf4]
          vfmaddsub132ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmaddsub132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0x96,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddsub132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmaddsub132ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0x96,0x31]
          vfmaddsub132ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmaddsub132ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0x96,0x71,0x7f]
          vfmaddsub132ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmaddsub132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0x96,0x72,0x80]
          vfmaddsub132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmaddsub213ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xa6,0xf4]
          vfmaddsub213ph %zmm28, %zmm29, %zmm30

// CHECK: vfmaddsub213ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xa6,0xf4]
          vfmaddsub213ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmaddsub213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xa6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddsub213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmaddsub213ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xa6,0x31]
          vfmaddsub213ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmaddsub213ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xa6,0x71,0x7f]
          vfmaddsub213ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmaddsub213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xa6,0x72,0x80]
          vfmaddsub213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmaddsub231ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xb6,0xf4]
          vfmaddsub231ph %zmm28, %zmm29, %zmm30

// CHECK: vfmaddsub231ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xb6,0xf4]
          vfmaddsub231ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmaddsub231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xb6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddsub231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmaddsub231ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xb6,0x31]
          vfmaddsub231ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmaddsub231ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xb6,0x71,0x7f]
          vfmaddsub231ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmaddsub231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xb6,0x72,0x80]
          vfmaddsub231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmsub132ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0x9a,0xf4]
          vfmsub132ph %zmm28, %zmm29, %zmm30

// CHECK: vfmsub132ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x9a,0xf4]
          vfmsub132ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmsub132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0x9a,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmsub132ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0x9a,0x31]
          vfmsub132ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmsub132ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0x9a,0x71,0x7f]
          vfmsub132ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmsub132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0x9a,0x72,0x80]
          vfmsub132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmsub132sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x9b,0xf4]
          vfmsub132sh %xmm28, %xmm29, %xmm30

// CHECK: vfmsub132sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x9b,0xf4]
          vfmsub132sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfmsub132sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x9b,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub132sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfmsub132sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0x9b,0x31]
          vfmsub132sh  (%r9), %xmm29, %xmm30

// CHECK: vfmsub132sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x9b,0x71,0x7f]
          vfmsub132sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfmsub132sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0x9b,0x72,0x80]
          vfmsub132sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfmsub213ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xaa,0xf4]
          vfmsub213ph %zmm28, %zmm29, %zmm30

// CHECK: vfmsub213ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xaa,0xf4]
          vfmsub213ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmsub213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xaa,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmsub213ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xaa,0x31]
          vfmsub213ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmsub213ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xaa,0x71,0x7f]
          vfmsub213ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmsub213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xaa,0x72,0x80]
          vfmsub213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmsub213sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xab,0xf4]
          vfmsub213sh %xmm28, %xmm29, %xmm30

// CHECK: vfmsub213sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xab,0xf4]
          vfmsub213sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfmsub213sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xab,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub213sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfmsub213sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0xab,0x31]
          vfmsub213sh  (%r9), %xmm29, %xmm30

// CHECK: vfmsub213sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xab,0x71,0x7f]
          vfmsub213sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfmsub213sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0xab,0x72,0x80]
          vfmsub213sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfmsub231ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xba,0xf4]
          vfmsub231ph %zmm28, %zmm29, %zmm30

// CHECK: vfmsub231ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xba,0xf4]
          vfmsub231ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmsub231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xba,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmsub231ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xba,0x31]
          vfmsub231ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmsub231ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xba,0x71,0x7f]
          vfmsub231ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmsub231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xba,0x72,0x80]
          vfmsub231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmsub231sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xbb,0xf4]
          vfmsub231sh %xmm28, %xmm29, %xmm30

// CHECK: vfmsub231sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xbb,0xf4]
          vfmsub231sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfmsub231sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xbb,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsub231sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfmsub231sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0xbb,0x31]
          vfmsub231sh  (%r9), %xmm29, %xmm30

// CHECK: vfmsub231sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xbb,0x71,0x7f]
          vfmsub231sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfmsub231sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0xbb,0x72,0x80]
          vfmsub231sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfmsubadd132ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0x97,0xf4]
          vfmsubadd132ph %zmm28, %zmm29, %zmm30

// CHECK: vfmsubadd132ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x97,0xf4]
          vfmsubadd132ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmsubadd132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0x97,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsubadd132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmsubadd132ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0x97,0x31]
          vfmsubadd132ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmsubadd132ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0x97,0x71,0x7f]
          vfmsubadd132ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmsubadd132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0x97,0x72,0x80]
          vfmsubadd132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmsubadd213ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xa7,0xf4]
          vfmsubadd213ph %zmm28, %zmm29, %zmm30

// CHECK: vfmsubadd213ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xa7,0xf4]
          vfmsubadd213ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmsubadd213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xa7,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsubadd213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmsubadd213ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xa7,0x31]
          vfmsubadd213ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmsubadd213ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xa7,0x71,0x7f]
          vfmsubadd213ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmsubadd213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xa7,0x72,0x80]
          vfmsubadd213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmsubadd231ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xb7,0xf4]
          vfmsubadd231ph %zmm28, %zmm29, %zmm30

// CHECK: vfmsubadd231ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xb7,0xf4]
          vfmsubadd231ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmsubadd231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xb7,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmsubadd231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmsubadd231ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xb7,0x31]
          vfmsubadd231ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfmsubadd231ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xb7,0x71,0x7f]
          vfmsubadd231ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmsubadd231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xb7,0x72,0x80]
          vfmsubadd231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfnmadd132ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0x9c,0xf4]
          vfnmadd132ph %zmm28, %zmm29, %zmm30

// CHECK: vfnmadd132ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x9c,0xf4]
          vfnmadd132ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfnmadd132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0x9c,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfnmadd132ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0x9c,0x31]
          vfnmadd132ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfnmadd132ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0x9c,0x71,0x7f]
          vfnmadd132ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfnmadd132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0x9c,0x72,0x80]
          vfnmadd132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfnmadd132sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x9d,0xf4]
          vfnmadd132sh %xmm28, %xmm29, %xmm30

// CHECK: vfnmadd132sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x9d,0xf4]
          vfnmadd132sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfnmadd132sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x9d,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd132sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfnmadd132sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0x9d,0x31]
          vfnmadd132sh  (%r9), %xmm29, %xmm30

// CHECK: vfnmadd132sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x9d,0x71,0x7f]
          vfnmadd132sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfnmadd132sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0x9d,0x72,0x80]
          vfnmadd132sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfnmadd213ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xac,0xf4]
          vfnmadd213ph %zmm28, %zmm29, %zmm30

// CHECK: vfnmadd213ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xac,0xf4]
          vfnmadd213ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfnmadd213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xac,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfnmadd213ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xac,0x31]
          vfnmadd213ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfnmadd213ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xac,0x71,0x7f]
          vfnmadd213ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfnmadd213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xac,0x72,0x80]
          vfnmadd213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfnmadd213sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xad,0xf4]
          vfnmadd213sh %xmm28, %xmm29, %xmm30

// CHECK: vfnmadd213sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xad,0xf4]
          vfnmadd213sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfnmadd213sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xad,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd213sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfnmadd213sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0xad,0x31]
          vfnmadd213sh  (%r9), %xmm29, %xmm30

// CHECK: vfnmadd213sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xad,0x71,0x7f]
          vfnmadd213sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfnmadd213sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0xad,0x72,0x80]
          vfnmadd213sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfnmadd231ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xbc,0xf4]
          vfnmadd231ph %zmm28, %zmm29, %zmm30

// CHECK: vfnmadd231ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xbc,0xf4]
          vfnmadd231ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfnmadd231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xbc,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfnmadd231ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xbc,0x31]
          vfnmadd231ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfnmadd231ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xbc,0x71,0x7f]
          vfnmadd231ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfnmadd231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xbc,0x72,0x80]
          vfnmadd231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfnmadd231sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xbd,0xf4]
          vfnmadd231sh %xmm28, %xmm29, %xmm30

// CHECK: vfnmadd231sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xbd,0xf4]
          vfnmadd231sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfnmadd231sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xbd,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmadd231sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfnmadd231sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0xbd,0x31]
          vfnmadd231sh  (%r9), %xmm29, %xmm30

// CHECK: vfnmadd231sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xbd,0x71,0x7f]
          vfnmadd231sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfnmadd231sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0xbd,0x72,0x80]
          vfnmadd231sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfnmsub132ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0x9e,0xf4]
          vfnmsub132ph %zmm28, %zmm29, %zmm30

// CHECK: vfnmsub132ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x9e,0xf4]
          vfnmsub132ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfnmsub132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0x9e,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfnmsub132ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0x9e,0x31]
          vfnmsub132ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfnmsub132ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0x9e,0x71,0x7f]
          vfnmsub132ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfnmsub132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0x9e,0x72,0x80]
          vfnmsub132ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfnmsub132sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0x9f,0xf4]
          vfnmsub132sh %xmm28, %xmm29, %xmm30

// CHECK: vfnmsub132sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0x9f,0xf4]
          vfnmsub132sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfnmsub132sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0x9f,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub132sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfnmsub132sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0x9f,0x31]
          vfnmsub132sh  (%r9), %xmm29, %xmm30

// CHECK: vfnmsub132sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0x9f,0x71,0x7f]
          vfnmsub132sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfnmsub132sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0x9f,0x72,0x80]
          vfnmsub132sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfnmsub213ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xae,0xf4]
          vfnmsub213ph %zmm28, %zmm29, %zmm30

// CHECK: vfnmsub213ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xae,0xf4]
          vfnmsub213ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfnmsub213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xae,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfnmsub213ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xae,0x31]
          vfnmsub213ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfnmsub213ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xae,0x71,0x7f]
          vfnmsub213ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfnmsub213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xae,0x72,0x80]
          vfnmsub213ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfnmsub213sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xaf,0xf4]
          vfnmsub213sh %xmm28, %xmm29, %xmm30

// CHECK: vfnmsub213sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xaf,0xf4]
          vfnmsub213sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfnmsub213sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xaf,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub213sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfnmsub213sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0xaf,0x31]
          vfnmsub213sh  (%r9), %xmm29, %xmm30

// CHECK: vfnmsub213sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xaf,0x71,0x7f]
          vfnmsub213sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfnmsub213sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0xaf,0x72,0x80]
          vfnmsub213sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfnmsub231ph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x40,0xbe,0xf4]
          vfnmsub231ph %zmm28, %zmm29, %zmm30

// CHECK: vfnmsub231ph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xbe,0xf4]
          vfnmsub231ph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfnmsub231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x47,0xbe,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231ph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfnmsub231ph  (%r9){1to32}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x15,0x50,0xbe,0x31]
          vfnmsub231ph  (%r9){1to32}, %zmm29, %zmm30

// CHECK: vfnmsub231ph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x15,0x40,0xbe,0x71,0x7f]
          vfnmsub231ph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfnmsub231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0xd7,0xbe,0x72,0x80]
          vfnmsub231ph  -256(%rdx){1to32}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfnmsub231sh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x00,0xbf,0xf4]
          vfnmsub231sh %xmm28, %xmm29, %xmm30

// CHECK: vfnmsub231sh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x15,0x10,0xbf,0xf4]
          vfnmsub231sh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfnmsub231sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x15,0x07,0xbf,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfnmsub231sh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfnmsub231sh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x15,0x00,0xbf,0x31]
          vfnmsub231sh  (%r9), %xmm29, %xmm30

// CHECK: vfnmsub231sh  254(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x15,0x00,0xbf,0x71,0x7f]
          vfnmsub231sh  254(%rcx), %xmm29, %xmm30

// CHECK: vfnmsub231sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x15,0x87,0xbf,0x72,0x80]
          vfnmsub231sh  -256(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfcmaddcph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x17,0x40,0x56,0xf4]
          vfcmaddcph %zmm28, %zmm29, %zmm30

// CHECK: vfcmaddcph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x17,0x10,0x56,0xf4]
          vfcmaddcph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfcmaddcph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x17,0x47,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfcmaddcph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfcmaddcph  (%r9){1to16}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x17,0x50,0x56,0x31]
          vfcmaddcph  (%r9){1to16}, %zmm29, %zmm30

// CHECK: vfcmaddcph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x17,0x40,0x56,0x71,0x7f]
          vfcmaddcph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfcmaddcph  -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x17,0xd7,0x56,0x72,0x80]
          vfcmaddcph  -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfcmaddcsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x17,0x00,0x57,0xf4]
          vfcmaddcsh %xmm28, %xmm29, %xmm30

// CHECK: vfcmaddcsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x17,0x10,0x57,0xf4]
          vfcmaddcsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfcmaddcsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x17,0x07,0x57,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfcmaddcsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfcmaddcsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x17,0x00,0x57,0x31]
          vfcmaddcsh  (%r9), %xmm29, %xmm30

// CHECK: vfcmaddcsh  508(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x17,0x00,0x57,0x71,0x7f]
          vfcmaddcsh  508(%rcx), %xmm29, %xmm30

// CHECK: vfcmaddcsh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x17,0x87,0x57,0x72,0x80]
          vfcmaddcsh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfcmulcph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x17,0x40,0xd6,0xf4]
          vfcmulcph %zmm28, %zmm29, %zmm30

// CHECK: vfcmulcph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x17,0x10,0xd6,0xf4]
          vfcmulcph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfcmulcph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x17,0x47,0xd6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfcmulcph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfcmulcph  (%r9){1to16}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x17,0x50,0xd6,0x31]
          vfcmulcph  (%r9){1to16}, %zmm29, %zmm30

// CHECK: vfcmulcph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x17,0x40,0xd6,0x71,0x7f]
          vfcmulcph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfcmulcph  -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x17,0xd7,0xd6,0x72,0x80]
          vfcmulcph  -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfcmulcsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x17,0x00,0xd7,0xf4]
          vfcmulcsh %xmm28, %xmm29, %xmm30

// CHECK: vfcmulcsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x17,0x10,0xd7,0xf4]
          vfcmulcsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfcmulcsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x17,0x07,0xd7,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfcmulcsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfcmulcsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x17,0x00,0xd7,0x31]
          vfcmulcsh  (%r9), %xmm29, %xmm30

// CHECK: vfcmulcsh  508(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x17,0x00,0xd7,0x71,0x7f]
          vfcmulcsh  508(%rcx), %xmm29, %xmm30

// CHECK: vfcmulcsh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x17,0x87,0xd7,0x72,0x80]
          vfcmulcsh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfmaddcph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x16,0x40,0x56,0xf4]
          vfmaddcph %zmm28, %zmm29, %zmm30

// CHECK: vfmaddcph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x16,0x10,0x56,0xf4]
          vfmaddcph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmaddcph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x16,0x47,0x56,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddcph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmaddcph  (%r9){1to16}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x16,0x50,0x56,0x31]
          vfmaddcph  (%r9){1to16}, %zmm29, %zmm30

// CHECK: vfmaddcph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x16,0x40,0x56,0x71,0x7f]
          vfmaddcph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmaddcph  -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x16,0xd7,0x56,0x72,0x80]
          vfmaddcph  -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmaddcsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x16,0x00,0x57,0xf4]
          vfmaddcsh %xmm28, %xmm29, %xmm30

// CHECK: vfmaddcsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x16,0x10,0x57,0xf4]
          vfmaddcsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfmaddcsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x16,0x07,0x57,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmaddcsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfmaddcsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x16,0x00,0x57,0x31]
          vfmaddcsh  (%r9), %xmm29, %xmm30

// CHECK: vfmaddcsh  508(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x16,0x00,0x57,0x71,0x7f]
          vfmaddcsh  508(%rcx), %xmm29, %xmm30

// CHECK: vfmaddcsh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x16,0x87,0x57,0x72,0x80]
          vfmaddcsh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}

// CHECK: vfmulcph %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x16,0x40,0xd6,0xf4]
          vfmulcph %zmm28, %zmm29, %zmm30

// CHECK: vfmulcph {rn-sae}, %zmm28, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x06,0x16,0x10,0xd6,0xf4]
          vfmulcph {rn-sae}, %zmm28, %zmm29, %zmm30

// CHECK: vfmulcph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x16,0x47,0xd6,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmulcph  268435456(%rbp,%r14,8), %zmm29, %zmm30 {%k7}

// CHECK: vfmulcph  (%r9){1to16}, %zmm29, %zmm30
// CHECK: encoding: [0x62,0x46,0x16,0x50,0xd6,0x31]
          vfmulcph  (%r9){1to16}, %zmm29, %zmm30

// CHECK: vfmulcph  8128(%rcx), %zmm29, %zmm30
// CHECK: encoding: [0x62,0x66,0x16,0x40,0xd6,0x71,0x7f]
          vfmulcph  8128(%rcx), %zmm29, %zmm30

// CHECK: vfmulcph  -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x16,0xd7,0xd6,0x72,0x80]
          vfmulcph  -512(%rdx){1to16}, %zmm29, %zmm30 {%k7} {z}

// CHECK: vfmulcsh %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x16,0x00,0xd7,0xf4]
          vfmulcsh %xmm28, %xmm29, %xmm30

// CHECK: vfmulcsh {rn-sae}, %xmm28, %xmm29, %xmm30
// CHECK: encoding: [0x62,0x06,0x16,0x10,0xd7,0xf4]
          vfmulcsh {rn-sae}, %xmm28, %xmm29, %xmm30

// CHECK: vfmulcsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}
// CHECK: encoding: [0x62,0x26,0x16,0x07,0xd7,0xb4,0xf5,0x00,0x00,0x00,0x10]
          vfmulcsh  268435456(%rbp,%r14,8), %xmm29, %xmm30 {%k7}

// CHECK: vfmulcsh  (%r9), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x46,0x16,0x00,0xd7,0x31]
          vfmulcsh  (%r9), %xmm29, %xmm30

// CHECK: vfmulcsh  508(%rcx), %xmm29, %xmm30
// CHECK: encoding: [0x62,0x66,0x16,0x00,0xd7,0x71,0x7f]
          vfmulcsh  508(%rcx), %xmm29, %xmm30

// CHECK: vfmulcsh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}
// CHECK: encoding: [0x62,0x66,0x16,0x87,0xd7,0x72,0x80]
          vfmulcsh  -512(%rdx), %xmm29, %xmm30 {%k7} {z}
