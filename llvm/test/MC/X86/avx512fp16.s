// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s  | FileCheck %s

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
