// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vaddss  %xmm4, %xmm6, %xmm2
// CHECK:  encoding: [0xc5,0xca,0x58,0xd4]
          vaddss  %xmm4, %xmm6, %xmm2

// CHECK: vmulss  %xmm4, %xmm6, %xmm2
// CHECK:  encoding: [0xc5,0xca,0x59,0xd4]
          vmulss  %xmm4, %xmm6, %xmm2

// CHECK: vsubss  %xmm4, %xmm6, %xmm2
// CHECK:  encoding: [0xc5,0xca,0x5c,0xd4]
          vsubss  %xmm4, %xmm6, %xmm2

// CHECK: vdivss  %xmm4, %xmm6, %xmm2
// CHECK:  encoding: [0xc5,0xca,0x5e,0xd4]
          vdivss  %xmm4, %xmm6, %xmm2

// CHECK: vaddsd  %xmm4, %xmm6, %xmm2
// CHECK:  encoding: [0xc5,0xcb,0x58,0xd4]
          vaddsd  %xmm4, %xmm6, %xmm2

// CHECK: vmulsd  %xmm4, %xmm6, %xmm2
// CHECK:  encoding: [0xc5,0xcb,0x59,0xd4]
          vmulsd  %xmm4, %xmm6, %xmm2

// CHECK: vsubsd  %xmm4, %xmm6, %xmm2
// CHECK:  encoding: [0xc5,0xcb,0x5c,0xd4]
          vsubsd  %xmm4, %xmm6, %xmm2

// CHECK: vdivsd  %xmm4, %xmm6, %xmm2
// CHECK:  encoding: [0xc5,0xcb,0x5e,0xd4]
          vdivsd  %xmm4, %xmm6, %xmm2

// CHECK: vaddss  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK:  encoding: [0xc5,0xea,0x58,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vaddss  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vsubss  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK:  encoding: [0xc5,0xea,0x5c,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vsubss  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vmulss  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK:  encoding: [0xc5,0xea,0x59,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vmulss  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vdivss  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK:  encoding: [0xc5,0xea,0x5e,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vdivss  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vaddsd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK:  encoding: [0xc5,0xeb,0x58,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vaddsd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vsubsd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK:  encoding: [0xc5,0xeb,0x5c,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vsubsd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vmulsd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK:  encoding: [0xc5,0xeb,0x59,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vmulsd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vdivsd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK:  encoding: [0xc5,0xeb,0x5e,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vdivsd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vaddps  %xmm4, %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc8,0x58,0xd4]
          vaddps  %xmm4, %xmm6, %xmm2

// CHECK: vsubps  %xmm4, %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc8,0x5c,0xd4]
          vsubps  %xmm4, %xmm6, %xmm2

// CHECK: vmulps  %xmm4, %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc8,0x59,0xd4]
          vmulps  %xmm4, %xmm6, %xmm2

// CHECK: vdivps  %xmm4, %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc8,0x5e,0xd4]
          vdivps  %xmm4, %xmm6, %xmm2

// CHECK: vaddpd  %xmm4, %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc9,0x58,0xd4]
          vaddpd  %xmm4, %xmm6, %xmm2

// CHECK: vsubpd  %xmm4, %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc9,0x5c,0xd4]
          vsubpd  %xmm4, %xmm6, %xmm2

// CHECK: vmulpd  %xmm4, %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc9,0x59,0xd4]
          vmulpd  %xmm4, %xmm6, %xmm2

// CHECK: vdivpd  %xmm4, %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc9,0x5e,0xd4]
          vdivpd  %xmm4, %xmm6, %xmm2

// CHECK: vaddps  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x58,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vaddps  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vsubps  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x5c,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vsubps  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vmulps  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x59,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vmulps  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vdivps  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x5e,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vdivps  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vaddpd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x58,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vaddpd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vsubpd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x5c,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vsubpd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vmulpd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x59,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vmulpd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vdivpd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x5e,0xac,0xcb,0xef,0xbe,0xad,0xde]
          vdivpd  3735928559(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: vmaxss  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xda,0x5f,0xf2]
          vmaxss  %xmm2, %xmm4, %xmm6

// CHECK: vmaxsd  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xdb,0x5f,0xf2]
          vmaxsd  %xmm2, %xmm4, %xmm6

// CHECK: vminss  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xda,0x5d,0xf2]
          vminss  %xmm2, %xmm4, %xmm6

// CHECK: vminsd  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xdb,0x5d,0xf2]
          vminsd  %xmm2, %xmm4, %xmm6

// CHECK: vmaxss  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xea,0x5f,0x6c,0xcb,0xfc]
          vmaxss  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vmaxsd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xeb,0x5f,0x6c,0xcb,0xfc]
          vmaxsd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vminss  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xea,0x5d,0x6c,0xcb,0xfc]
          vminss  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vminsd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xeb,0x5d,0x6c,0xcb,0xfc]
          vminsd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vmaxps  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd8,0x5f,0xf2]
          vmaxps  %xmm2, %xmm4, %xmm6

// CHECK: vmaxpd  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd9,0x5f,0xf2]
          vmaxpd  %xmm2, %xmm4, %xmm6

// CHECK: vminps  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd8,0x5d,0xf2]
          vminps  %xmm2, %xmm4, %xmm6

// CHECK: vminpd  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd9,0x5d,0xf2]
          vminpd  %xmm2, %xmm4, %xmm6

// CHECK: vmaxps  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x5f,0x6c,0xcb,0xfc]
          vmaxps  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vmaxpd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x5f,0x6c,0xcb,0xfc]
          vmaxpd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vminps  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x5d,0x6c,0xcb,0xfc]
          vminps  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vminpd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x5d,0x6c,0xcb,0xfc]
          vminpd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vandps  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd8,0x54,0xf2]
          vandps  %xmm2, %xmm4, %xmm6

// CHECK: vandpd  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd9,0x54,0xf2]
          vandpd  %xmm2, %xmm4, %xmm6

// CHECK: vandps  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x54,0x6c,0xcb,0xfc]
          vandps  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vandpd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x54,0x6c,0xcb,0xfc]
          vandpd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vorps  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd8,0x56,0xf2]
          vorps  %xmm2, %xmm4, %xmm6

// CHECK: vorpd  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd9,0x56,0xf2]
          vorpd  %xmm2, %xmm4, %xmm6

// CHECK: vorps  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x56,0x6c,0xcb,0xfc]
          vorps  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vorpd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x56,0x6c,0xcb,0xfc]
          vorpd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vxorps  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd8,0x57,0xf2]
          vxorps  %xmm2, %xmm4, %xmm6

// CHECK: vxorpd  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd9,0x57,0xf2]
          vxorpd  %xmm2, %xmm4, %xmm6

// CHECK: vxorps  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x57,0x6c,0xcb,0xfc]
          vxorps  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vxorpd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x57,0x6c,0xcb,0xfc]
          vxorpd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vandnps  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd8,0x55,0xf2]
          vandnps  %xmm2, %xmm4, %xmm6

// CHECK: vandnpd  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd9,0x55,0xf2]
          vandnpd  %xmm2, %xmm4, %xmm6

// CHECK: vandnps  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x55,0x6c,0xcb,0xfc]
          vandnps  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vandnpd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x55,0x6c,0xcb,0xfc]
          vandnpd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vmovss  -4(%ebx,%ecx,8), %xmm5
// CHECK: encoding: [0xc5,0xfa,0x10,0x6c,0xcb,0xfc]
          vmovss  -4(%ebx,%ecx,8), %xmm5

// CHECK: vmovss  %xmm4, %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xea,0x10,0xec]
          vmovss  %xmm4, %xmm2, %xmm5

// CHECK: vmovsd  -4(%ebx,%ecx,8), %xmm5
// CHECK: encoding: [0xc5,0xfb,0x10,0x6c,0xcb,0xfc]
          vmovsd  -4(%ebx,%ecx,8), %xmm5

// CHECK: vmovsd  %xmm4, %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xeb,0x10,0xec]
          vmovsd  %xmm4, %xmm2, %xmm5

// CHECK: vunpckhps  %xmm1, %xmm2, %xmm4
// CHECK: encoding: [0xc5,0xe8,0x15,0xe1]
          vunpckhps  %xmm1, %xmm2, %xmm4

// CHECK: vunpckhpd  %xmm1, %xmm2, %xmm4
// CHECK: encoding: [0xc5,0xe9,0x15,0xe1]
          vunpckhpd  %xmm1, %xmm2, %xmm4

// CHECK: vunpcklps  %xmm1, %xmm2, %xmm4
// CHECK: encoding: [0xc5,0xe8,0x14,0xe1]
          vunpcklps  %xmm1, %xmm2, %xmm4

// CHECK: vunpcklpd  %xmm1, %xmm2, %xmm4
// CHECK: encoding: [0xc5,0xe9,0x14,0xe1]
          vunpcklpd  %xmm1, %xmm2, %xmm4

// CHECK: vunpckhps  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x15,0x6c,0xcb,0xfc]
          vunpckhps  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vunpckhpd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x15,0x6c,0xcb,0xfc]
          vunpckhpd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vunpcklps  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe8,0x14,0x6c,0xcb,0xfc]
          vunpcklps  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vunpcklpd  -4(%ebx,%ecx,8), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0x14,0x6c,0xcb,0xfc]
          vunpcklpd  -4(%ebx,%ecx,8), %xmm2, %xmm5

// CHECK: vcmpps  $0, %xmm0, %xmm6, %xmm1
// CHECK: encoding: [0xc5,0xc8,0xc2,0xc8,0x00]
          vcmpps  $0, %xmm0, %xmm6, %xmm1

// CHECK: vcmpps  $0, (%eax), %xmm6, %xmm1
// CHECK: encoding: [0xc5,0xc8,0xc2,0x08,0x00]
          vcmpps  $0, (%eax), %xmm6, %xmm1

// CHECK: vcmpps  $7, %xmm0, %xmm6, %xmm1
// CHECK: encoding: [0xc5,0xc8,0xc2,0xc8,0x07]
          vcmpps  $7, %xmm0, %xmm6, %xmm1

// CHECK: vcmppd  $0, %xmm0, %xmm6, %xmm1
// CHECK: encoding: [0xc5,0xc9,0xc2,0xc8,0x00]
          vcmppd  $0, %xmm0, %xmm6, %xmm1

// CHECK: vcmppd  $0, (%eax), %xmm6, %xmm1
// CHECK: encoding: [0xc5,0xc9,0xc2,0x08,0x00]
          vcmppd  $0, (%eax), %xmm6, %xmm1

// CHECK: vcmppd  $7, %xmm0, %xmm6, %xmm1
// CHECK: encoding: [0xc5,0xc9,0xc2,0xc8,0x07]
          vcmppd  $7, %xmm0, %xmm6, %xmm1

// CHECK: vshufps  $8, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc6,0xd9,0x08]
          vshufps  $8, %xmm1, %xmm2, %xmm3

// CHECK: vshufps  $8, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc6,0x5c,0xcb,0xfc,0x08]
          vshufps  $8, -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vshufpd  $8, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc6,0xd9,0x08]
          vshufpd  $8, %xmm1, %xmm2, %xmm3

// CHECK: vshufpd  $8, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc6,0x5c,0xcb,0xfc,0x08]
          vshufpd  $8, -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpps  $0, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x00]
          vcmpeqps   %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $2, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x02]
          vcmpleps   %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $1, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x01]
          vcmpltps   %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $4, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x04]
          vcmpneqps   %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $6, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x06]
          vcmpnleps   %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $5, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x05]
          vcmpnltps   %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $7, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x07]
          vcmpordps   %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $3, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x03]
          vcmpunordps   %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $0, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0x5c,0xcb,0xfc,0x00]
          vcmpeqps   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpps  $2, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0x5c,0xcb,0xfc,0x02]
          vcmpleps   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpps  $1, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0x5c,0xcb,0xfc,0x01]
          vcmpltps   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpps  $4, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0x5c,0xcb,0xfc,0x04]
          vcmpneqps   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpps  $6, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0x5c,0xcb,0xfc,0x06]
          vcmpnleps   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpps  $5, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0x5c,0xcb,0xfc,0x05]
          vcmpnltps   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpps  $7, -4(%ebx,%ecx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc8,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordps   -4(%ebx,%ecx,8), %xmm6, %xmm2

// CHECK: vcmpps  $3, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0x5c,0xcb,0xfc,0x03]
          vcmpunordps   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmppd  $0, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0xd9,0x00]
          vcmpeqpd   %xmm1, %xmm2, %xmm3

// CHECK: vcmppd  $2, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0xd9,0x02]
          vcmplepd   %xmm1, %xmm2, %xmm3

// CHECK: vcmppd  $1, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0xd9,0x01]
          vcmpltpd   %xmm1, %xmm2, %xmm3

// CHECK: vcmppd  $4, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0xd9,0x04]
          vcmpneqpd   %xmm1, %xmm2, %xmm3

// CHECK: vcmppd  $6, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0xd9,0x06]
          vcmpnlepd   %xmm1, %xmm2, %xmm3

// CHECK: vcmppd  $5, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0xd9,0x05]
          vcmpnltpd   %xmm1, %xmm2, %xmm3

// CHECK: vcmppd  $7, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0xd9,0x07]
          vcmpordpd   %xmm1, %xmm2, %xmm3

// CHECK: vcmppd  $3, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0xd9,0x03]
          vcmpunordpd   %xmm1, %xmm2, %xmm3

// CHECK: vcmppd  $0, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0x5c,0xcb,0xfc,0x00]
          vcmpeqpd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmppd  $2, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0x5c,0xcb,0xfc,0x02]
          vcmplepd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmppd  $1, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0x5c,0xcb,0xfc,0x01]
          vcmpltpd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmppd  $4, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0x5c,0xcb,0xfc,0x04]
          vcmpneqpd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmppd  $6, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0x5c,0xcb,0xfc,0x06]
          vcmpnlepd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmppd  $5, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0x5c,0xcb,0xfc,0x05]
          vcmpnltpd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmppd  $7, -4(%ebx,%ecx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc9,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordpd   -4(%ebx,%ecx,8), %xmm6, %xmm2

// CHECK: vcmppd  $3, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc2,0x5c,0xcb,0xfc,0x03]
          vcmpunordpd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vmovmskps  %xmm2, %eax
// CHECK: encoding: [0xc5,0xf8,0x50,0xc2]
          vmovmskps  %xmm2, %eax

// CHECK: vmovmskpd  %xmm2, %eax
// CHECK: encoding: [0xc5,0xf9,0x50,0xc2]
          vmovmskpd  %xmm2, %eax

// CHECK: vmovmskps  %ymm2, %eax
// CHECK: encoding: [0xc5,0xfc,0x50,0xc2]
          vmovmskps  %ymm2, %eax

// CHECK: vmovmskpd  %ymm2, %eax
// CHECK: encoding: [0xc5,0xfd,0x50,0xc2]
          vmovmskpd  %ymm2, %eax

// CHECK: vcmpss  $0, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0xd9,0x00]
          vcmpeqss   %xmm1, %xmm2, %xmm3

// CHECK: vcmpss  $2, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0xd9,0x02]
          vcmpless   %xmm1, %xmm2, %xmm3

// CHECK: vcmpss  $1, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0xd9,0x01]
          vcmpltss   %xmm1, %xmm2, %xmm3

// CHECK: vcmpss  $4, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0xd9,0x04]
          vcmpneqss   %xmm1, %xmm2, %xmm3

// CHECK: vcmpss  $6, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0xd9,0x06]
          vcmpnless   %xmm1, %xmm2, %xmm3

// CHECK: vcmpss  $5, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0xd9,0x05]
          vcmpnltss   %xmm1, %xmm2, %xmm3

// CHECK: vcmpss  $7, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0xd9,0x07]
          vcmpordss   %xmm1, %xmm2, %xmm3

// CHECK: vcmpss  $3, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0xd9,0x03]
          vcmpunordss   %xmm1, %xmm2, %xmm3

// CHECK: vcmpss  $0, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0x5c,0xcb,0xfc,0x00]
          vcmpeqss   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpss  $2, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0x5c,0xcb,0xfc,0x02]
          vcmpless   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpss  $1, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0x5c,0xcb,0xfc,0x01]
          vcmpltss   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpss  $4, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0x5c,0xcb,0xfc,0x04]
          vcmpneqss   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpss  $6, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0x5c,0xcb,0xfc,0x06]
          vcmpnless   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpss  $5, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0x5c,0xcb,0xfc,0x05]
          vcmpnltss   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpss  $7, -4(%ebx,%ecx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xca,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordss   -4(%ebx,%ecx,8), %xmm6, %xmm2

// CHECK: vcmpss  $3, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0xc2,0x5c,0xcb,0xfc,0x03]
          vcmpunordss   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpsd  $0, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0xd9,0x00]
          vcmpeqsd   %xmm1, %xmm2, %xmm3

// CHECK: vcmpsd  $2, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0xd9,0x02]
          vcmplesd   %xmm1, %xmm2, %xmm3

// CHECK: vcmpsd  $1, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0xd9,0x01]
          vcmpltsd   %xmm1, %xmm2, %xmm3

// CHECK: vcmpsd  $4, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0xd9,0x04]
          vcmpneqsd   %xmm1, %xmm2, %xmm3

// CHECK: vcmpsd  $6, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0xd9,0x06]
          vcmpnlesd   %xmm1, %xmm2, %xmm3

// CHECK: vcmpsd  $5, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0xd9,0x05]
          vcmpnltsd   %xmm1, %xmm2, %xmm3

// CHECK: vcmpsd  $7, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0xd9,0x07]
          vcmpordsd   %xmm1, %xmm2, %xmm3

// CHECK: vcmpsd  $3, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0xd9,0x03]
          vcmpunordsd   %xmm1, %xmm2, %xmm3

// CHECK: vcmpsd  $0, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0x5c,0xcb,0xfc,0x00]
          vcmpeqsd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpsd  $2, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0x5c,0xcb,0xfc,0x02]
          vcmplesd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpsd  $1, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0x5c,0xcb,0xfc,0x01]
          vcmpltsd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpsd  $4, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0x5c,0xcb,0xfc,0x04]
          vcmpneqsd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpsd  $6, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0x5c,0xcb,0xfc,0x06]
          vcmpnlesd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpsd  $5, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0x5c,0xcb,0xfc,0x05]
          vcmpnltsd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vcmpsd  $7, -4(%ebx,%ecx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xcb,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordsd   -4(%ebx,%ecx,8), %xmm6, %xmm2

// CHECK: vcmpsd  $3, -4(%ebx,%ecx,8), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xc2,0x5c,0xcb,0xfc,0x03]
          vcmpunordsd   -4(%ebx,%ecx,8), %xmm2, %xmm3

// CHECK: vucomiss  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf8,0x2e,0xd1]
          vucomiss  %xmm1, %xmm2

// CHECK: vucomiss  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf8,0x2e,0x10]
          vucomiss  (%eax), %xmm2

// CHECK: vcomiss  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf8,0x2f,0xd1]
          vcomiss  %xmm1, %xmm2

// CHECK: vcomiss  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf8,0x2f,0x10]
          vcomiss  (%eax), %xmm2

// CHECK: vucomisd  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf9,0x2e,0xd1]
          vucomisd  %xmm1, %xmm2

// CHECK: vucomisd  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf9,0x2e,0x10]
          vucomisd  (%eax), %xmm2

// CHECK: vcomisd  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf9,0x2f,0xd1]
          vcomisd  %xmm1, %xmm2

// CHECK: vcomisd  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf9,0x2f,0x10]
          vcomisd  (%eax), %xmm2

// CHECK: vcvttss2si  %xmm1, %eax
// CHECK: encoding: [0xc5,0xfa,0x2c,0xc1]
          vcvttss2si  %xmm1, %eax

// CHECK: vcvttss2si  (%ecx), %eax
// CHECK: encoding: [0xc5,0xfa,0x2c,0x01]
          vcvttss2si  (%ecx), %eax

// CHECK: vcvtsi2ss  (%eax), %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf2,0x2a,0x10]
          vcvtsi2ss  (%eax), %xmm1, %xmm2

// CHECK: vcvtsi2ss  (%eax), %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf2,0x2a,0x10]
          vcvtsi2ss  (%eax), %xmm1, %xmm2

// CHECK: vcvttsd2si  %xmm1, %eax
// CHECK: encoding: [0xc5,0xfb,0x2c,0xc1]
          vcvttsd2si  %xmm1, %eax

// CHECK: vcvttsd2si  (%ecx), %eax
// CHECK: encoding: [0xc5,0xfb,0x2c,0x01]
          vcvttsd2si  (%ecx), %eax

// CHECK: vcvtsi2sd  (%eax), %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf3,0x2a,0x10]
          vcvtsi2sd  (%eax), %xmm1, %xmm2

// CHECK: vcvtsi2sd  (%eax), %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf3,0x2a,0x10]
          vcvtsi2sd  (%eax), %xmm1, %xmm2

// CHECK: vmovaps  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf8,0x28,0x10]
          vmovaps  (%eax), %xmm2

// CHECK: vmovaps  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf8,0x28,0xd1]
          vmovaps  %xmm1, %xmm2

// CHECK: vmovaps  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf8,0x29,0x08]
          vmovaps  %xmm1, (%eax)

// CHECK: vmovapd  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf9,0x28,0x10]
          vmovapd  (%eax), %xmm2

// CHECK: vmovapd  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf9,0x28,0xd1]
          vmovapd  %xmm1, %xmm2

// CHECK: vmovapd  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf9,0x29,0x08]
          vmovapd  %xmm1, (%eax)

// CHECK: vmovups  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf8,0x10,0x10]
          vmovups  (%eax), %xmm2

// CHECK: vmovups  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf8,0x10,0xd1]
          vmovups  %xmm1, %xmm2

// CHECK: vmovups  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf8,0x11,0x08]
          vmovups  %xmm1, (%eax)

// CHECK: vmovupd  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf9,0x10,0x10]
          vmovupd  (%eax), %xmm2

// CHECK: vmovupd  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf9,0x10,0xd1]
          vmovupd  %xmm1, %xmm2

// CHECK: vmovupd  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf9,0x11,0x08]
          vmovupd  %xmm1, (%eax)

// CHECK: vmovlps  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf8,0x13,0x08]
          vmovlps  %xmm1, (%eax)

// CHECK: vmovlps  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0x12,0x18]
          vmovlps  (%eax), %xmm2, %xmm3

// CHECK: vmovlpd  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf9,0x13,0x08]
          vmovlpd  %xmm1, (%eax)

// CHECK: vmovlpd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x12,0x18]
          vmovlpd  (%eax), %xmm2, %xmm3

// CHECK: vmovhps  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf8,0x17,0x08]
          vmovhps  %xmm1, (%eax)

// CHECK: vmovhps  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0x16,0x18]
          vmovhps  (%eax), %xmm2, %xmm3

// CHECK: vmovhpd  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf9,0x17,0x08]
          vmovhpd  %xmm1, (%eax)

// CHECK: vmovhpd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x16,0x18]
          vmovhpd  (%eax), %xmm2, %xmm3

// CHECK: vmovlhps  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0x16,0xd9]
          vmovlhps  %xmm1, %xmm2, %xmm3

// CHECK: vmovhlps  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0x12,0xd9]
          vmovhlps  %xmm1, %xmm2, %xmm3

// CHECK: vcvtss2sil  %xmm1, %eax
// CHECK: encoding: [0xc5,0xfa,0x2d,0xc1]
          vcvtss2si  %xmm1, %eax

// CHECK: vcvtss2sil  (%eax), %ebx
// CHECK: encoding: [0xc5,0xfa,0x2d,0x18]
          vcvtss2si  (%eax), %ebx

// CHECK: vcvtdq2ps  %xmm5, %xmm6
// CHECK: encoding: [0xc5,0xf8,0x5b,0xf5]
          vcvtdq2ps  %xmm5, %xmm6

// CHECK: vcvtdq2ps  (%eax), %xmm6
// CHECK: encoding: [0xc5,0xf8,0x5b,0x30]
          vcvtdq2ps  (%eax), %xmm6

// CHECK: vcvtsd2ss  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xdb,0x5a,0xf2]
          vcvtsd2ss  %xmm2, %xmm4, %xmm6

// CHECK: vcvtsd2ss  (%eax), %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xdb,0x5a,0x30]
          vcvtsd2ss  (%eax), %xmm4, %xmm6

// CHECK: vcvtps2dq  %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xf9,0x5b,0xda]
          vcvtps2dq  %xmm2, %xmm3

// CHECK: vcvtps2dq  (%eax), %xmm3
// CHECK: encoding: [0xc5,0xf9,0x5b,0x18]
          vcvtps2dq  (%eax), %xmm3

// CHECK: vcvtss2sd  %xmm2, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xda,0x5a,0xf2]
          vcvtss2sd  %xmm2, %xmm4, %xmm6

// CHECK: vcvtss2sd  (%eax), %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xda,0x5a,0x30]
          vcvtss2sd  (%eax), %xmm4, %xmm6

// CHECK: vcvtdq2ps  %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xf8,0x5b,0xf4]
          vcvtdq2ps  %xmm4, %xmm6

// CHECK: vcvtdq2ps  (%ecx), %xmm4
// CHECK: encoding: [0xc5,0xf8,0x5b,0x21]
          vcvtdq2ps  (%ecx), %xmm4

// CHECK: vcvttps2dq  %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xfa,0x5b,0xda]
          vcvttps2dq  %xmm2, %xmm3

// CHECK: vcvttps2dq  (%eax), %xmm3
// CHECK: encoding: [0xc5,0xfa,0x5b,0x18]
          vcvttps2dq  (%eax), %xmm3

// CHECK: vcvtps2pd  %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xf8,0x5a,0xda]
          vcvtps2pd  %xmm2, %xmm3

// CHECK: vcvtps2pd  (%eax), %xmm3
// CHECK: encoding: [0xc5,0xf8,0x5a,0x18]
          vcvtps2pd  (%eax), %xmm3

// CHECK: vcvtpd2ps  %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xf9,0x5a,0xda]
          vcvtpd2ps  %xmm2, %xmm3

// CHECK: vsqrtpd  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf9,0x51,0xd1]
          vsqrtpd  %xmm1, %xmm2

// CHECK: vsqrtpd  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf9,0x51,0x10]
          vsqrtpd  (%eax), %xmm2

// CHECK: vsqrtps  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf8,0x51,0xd1]
          vsqrtps  %xmm1, %xmm2

// CHECK: vsqrtps  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf8,0x51,0x10]
          vsqrtps  (%eax), %xmm2

// CHECK: vsqrtsd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0x51,0xd9]
          vsqrtsd  %xmm1, %xmm2, %xmm3

// CHECK: vsqrtsd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0x51,0x18]
          vsqrtsd  (%eax), %xmm2, %xmm3

// CHECK: vsqrtss  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0x51,0xd9]
          vsqrtss  %xmm1, %xmm2, %xmm3

// CHECK: vsqrtss  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0x51,0x18]
          vsqrtss  (%eax), %xmm2, %xmm3

// CHECK: vrsqrtps  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf8,0x52,0xd1]
          vrsqrtps  %xmm1, %xmm2

// CHECK: vrsqrtps  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf8,0x52,0x10]
          vrsqrtps  (%eax), %xmm2

// CHECK: vrsqrtss  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0x52,0xd9]
          vrsqrtss  %xmm1, %xmm2, %xmm3

// CHECK: vrsqrtss  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0x52,0x18]
          vrsqrtss  (%eax), %xmm2, %xmm3

// CHECK: vrcpps  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf8,0x53,0xd1]
          vrcpps  %xmm1, %xmm2

// CHECK: vrcpps  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xf8,0x53,0x10]
          vrcpps  (%eax), %xmm2

// CHECK: vrcpss  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0x53,0xd9]
          vrcpss  %xmm1, %xmm2, %xmm3

// CHECK: vrcpss  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xea,0x53,0x18]
          vrcpss  (%eax), %xmm2, %xmm3

// CHECK: vmovntdq  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf9,0xe7,0x08]
          vmovntdq  %xmm1, (%eax)

// CHECK: vmovntpd  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf9,0x2b,0x08]
          vmovntpd  %xmm1, (%eax)

// CHECK: vmovntps  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf8,0x2b,0x08]
          vmovntps  %xmm1, (%eax)

// CHECK: vldmxcsr  (%eax)
// CHECK: encoding: [0xc5,0xf8,0xae,0x10]
          vldmxcsr  (%eax)

// CHECK: vstmxcsr  (%eax)
// CHECK: encoding: [0xc5,0xf8,0xae,0x18]
          vstmxcsr  (%eax)

// CHECK: vldmxcsr  3735928559
// CHECK: encoding: [0xc5,0xf8,0xae,0x15,0xef,0xbe,0xad,0xde]
          vldmxcsr  0xdeadbeef

// CHECK: vstmxcsr  3735928559
// CHECK: encoding: [0xc5,0xf8,0xae,0x1d,0xef,0xbe,0xad,0xde]
          vstmxcsr  0xdeadbeef

// CHECK: vpsubb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf8,0xd9]
          vpsubb  %xmm1, %xmm2, %xmm3

// CHECK: vpsubb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf8,0x18]
          vpsubb  (%eax), %xmm2, %xmm3

// CHECK: vpsubw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf9,0xd9]
          vpsubw  %xmm1, %xmm2, %xmm3

// CHECK: vpsubw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf9,0x18]
          vpsubw  (%eax), %xmm2, %xmm3

// CHECK: vpsubd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfa,0xd9]
          vpsubd  %xmm1, %xmm2, %xmm3

// CHECK: vpsubd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfa,0x18]
          vpsubd  (%eax), %xmm2, %xmm3

// CHECK: vpsubq  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfb,0xd9]
          vpsubq  %xmm1, %xmm2, %xmm3

// CHECK: vpsubq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfb,0x18]
          vpsubq  (%eax), %xmm2, %xmm3

// CHECK: vpsubsb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe8,0xd9]
          vpsubsb  %xmm1, %xmm2, %xmm3

// CHECK: vpsubsb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe8,0x18]
          vpsubsb  (%eax), %xmm2, %xmm3

// CHECK: vpsubsw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe9,0xd9]
          vpsubsw  %xmm1, %xmm2, %xmm3

// CHECK: vpsubsw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe9,0x18]
          vpsubsw  (%eax), %xmm2, %xmm3

// CHECK: vpsubusb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd8,0xd9]
          vpsubusb  %xmm1, %xmm2, %xmm3

// CHECK: vpsubusb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd8,0x18]
          vpsubusb  (%eax), %xmm2, %xmm3

// CHECK: vpsubusw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd9,0xd9]
          vpsubusw  %xmm1, %xmm2, %xmm3

// CHECK: vpsubusw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd9,0x18]
          vpsubusw  (%eax), %xmm2, %xmm3

// CHECK: vpaddb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfc,0xd9]
          vpaddb  %xmm1, %xmm2, %xmm3

// CHECK: vpaddb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfc,0x18]
          vpaddb  (%eax), %xmm2, %xmm3

// CHECK: vpaddw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfd,0xd9]
          vpaddw  %xmm1, %xmm2, %xmm3

// CHECK: vpaddw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfd,0x18]
          vpaddw  (%eax), %xmm2, %xmm3

// CHECK: vpaddd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfe,0xd9]
          vpaddd  %xmm1, %xmm2, %xmm3

// CHECK: vpaddd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xfe,0x18]
          vpaddd  (%eax), %xmm2, %xmm3

// CHECK: vpaddq  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd4,0xd9]
          vpaddq  %xmm1, %xmm2, %xmm3

// CHECK: vpaddq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd4,0x18]
          vpaddq  (%eax), %xmm2, %xmm3

// CHECK: vpaddsb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xec,0xd9]
          vpaddsb  %xmm1, %xmm2, %xmm3

// CHECK: vpaddsb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xec,0x18]
          vpaddsb  (%eax), %xmm2, %xmm3

// CHECK: vpaddsw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xed,0xd9]
          vpaddsw  %xmm1, %xmm2, %xmm3

// CHECK: vpaddsw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xed,0x18]
          vpaddsw  (%eax), %xmm2, %xmm3

// CHECK: vpaddusb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xdc,0xd9]
          vpaddusb  %xmm1, %xmm2, %xmm3

// CHECK: vpaddusb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xdc,0x18]
          vpaddusb  (%eax), %xmm2, %xmm3

// CHECK: vpaddusw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xdd,0xd9]
          vpaddusw  %xmm1, %xmm2, %xmm3

// CHECK: vpaddusw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xdd,0x18]
          vpaddusw  (%eax), %xmm2, %xmm3

// CHECK: vpmulhuw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe4,0xd9]
          vpmulhuw  %xmm1, %xmm2, %xmm3

// CHECK: vpmulhuw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe4,0x18]
          vpmulhuw  (%eax), %xmm2, %xmm3

// CHECK: vpmulhw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe5,0xd9]
          vpmulhw  %xmm1, %xmm2, %xmm3

// CHECK: vpmulhw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe5,0x18]
          vpmulhw  (%eax), %xmm2, %xmm3

// CHECK: vpmullw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd5,0xd9]
          vpmullw  %xmm1, %xmm2, %xmm3

// CHECK: vpmullw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd5,0x18]
          vpmullw  (%eax), %xmm2, %xmm3

// CHECK: vpmuludq  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf4,0xd9]
          vpmuludq  %xmm1, %xmm2, %xmm3

// CHECK: vpmuludq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf4,0x18]
          vpmuludq  (%eax), %xmm2, %xmm3

// CHECK: vpavgb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe0,0xd9]
          vpavgb  %xmm1, %xmm2, %xmm3

// CHECK: vpavgb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe0,0x18]
          vpavgb  (%eax), %xmm2, %xmm3

// CHECK: vpavgw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe3,0xd9]
          vpavgw  %xmm1, %xmm2, %xmm3

// CHECK: vpavgw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe3,0x18]
          vpavgw  (%eax), %xmm2, %xmm3

// CHECK: vpminsw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xea,0xd9]
          vpminsw  %xmm1, %xmm2, %xmm3

// CHECK: vpminsw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xea,0x18]
          vpminsw  (%eax), %xmm2, %xmm3

// CHECK: vpminub  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xda,0xd9]
          vpminub  %xmm1, %xmm2, %xmm3

// CHECK: vpminub  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xda,0x18]
          vpminub  (%eax), %xmm2, %xmm3

// CHECK: vpmaxsw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xee,0xd9]
          vpmaxsw  %xmm1, %xmm2, %xmm3

// CHECK: vpmaxsw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xee,0x18]
          vpmaxsw  (%eax), %xmm2, %xmm3

// CHECK: vpmaxub  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xde,0xd9]
          vpmaxub  %xmm1, %xmm2, %xmm3

// CHECK: vpmaxub  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xde,0x18]
          vpmaxub  (%eax), %xmm2, %xmm3

// CHECK: vpsadbw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf6,0xd9]
          vpsadbw  %xmm1, %xmm2, %xmm3

// CHECK: vpsadbw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf6,0x18]
          vpsadbw  (%eax), %xmm2, %xmm3

// CHECK: vpsllw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf1,0xd9]
          vpsllw  %xmm1, %xmm2, %xmm3

// CHECK: vpsllw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf1,0x18]
          vpsllw  (%eax), %xmm2, %xmm3

// CHECK: vpslld  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf2,0xd9]
          vpslld  %xmm1, %xmm2, %xmm3

// CHECK: vpslld  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf2,0x18]
          vpslld  (%eax), %xmm2, %xmm3

// CHECK: vpsllq  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf3,0xd9]
          vpsllq  %xmm1, %xmm2, %xmm3

// CHECK: vpsllq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xf3,0x18]
          vpsllq  (%eax), %xmm2, %xmm3

// CHECK: vpsraw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe1,0xd9]
          vpsraw  %xmm1, %xmm2, %xmm3

// CHECK: vpsraw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe1,0x18]
          vpsraw  (%eax), %xmm2, %xmm3

// CHECK: vpsrad  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe2,0xd9]
          vpsrad  %xmm1, %xmm2, %xmm3

// CHECK: vpsrad  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xe2,0x18]
          vpsrad  (%eax), %xmm2, %xmm3

// CHECK: vpsrlw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd1,0xd9]
          vpsrlw  %xmm1, %xmm2, %xmm3

// CHECK: vpsrlw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd1,0x18]
          vpsrlw  (%eax), %xmm2, %xmm3

// CHECK: vpsrld  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd2,0xd9]
          vpsrld  %xmm1, %xmm2, %xmm3

// CHECK: vpsrld  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd2,0x18]
          vpsrld  (%eax), %xmm2, %xmm3

// CHECK: vpsrlq  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd3,0xd9]
          vpsrlq  %xmm1, %xmm2, %xmm3

// CHECK: vpsrlq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd3,0x18]
          vpsrlq  (%eax), %xmm2, %xmm3

// CHECK: vpslld  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x72,0xf2,0x0a]
          vpslld  $10, %xmm2, %xmm3

// CHECK: vpslldq  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x73,0xfa,0x0a]
          vpslldq  $10, %xmm2, %xmm3

// CHECK: vpsllq  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x73,0xf2,0x0a]
          vpsllq  $10, %xmm2, %xmm3

// CHECK: vpsllw  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x71,0xf2,0x0a]
          vpsllw  $10, %xmm2, %xmm3

// CHECK: vpsrad  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x72,0xe2,0x0a]
          vpsrad  $10, %xmm2, %xmm3

// CHECK: vpsraw  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x71,0xe2,0x0a]
          vpsraw  $10, %xmm2, %xmm3

// CHECK: vpsrld  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x72,0xd2,0x0a]
          vpsrld  $10, %xmm2, %xmm3

// CHECK: vpsrldq  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x73,0xda,0x0a]
          vpsrldq  $10, %xmm2, %xmm3

// CHECK: vpsrlq  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x73,0xd2,0x0a]
          vpsrlq  $10, %xmm2, %xmm3

// CHECK: vpsrlw  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x71,0xd2,0x0a]
          vpsrlw  $10, %xmm2, %xmm3

// CHECK: vpslld  $10, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe1,0x72,0xf2,0x0a]
          vpslld  $10, %xmm2, %xmm3

// CHECK: vpand  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xdb,0xd9]
          vpand  %xmm1, %xmm2, %xmm3

// CHECK: vpand  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xdb,0x18]
          vpand  (%eax), %xmm2, %xmm3

// CHECK: vpor  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xeb,0xd9]
          vpor  %xmm1, %xmm2, %xmm3

// CHECK: vpor  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xeb,0x18]
          vpor  (%eax), %xmm2, %xmm3

// CHECK: vpxor  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xef,0xd9]
          vpxor  %xmm1, %xmm2, %xmm3

// CHECK: vpxor  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xef,0x18]
          vpxor  (%eax), %xmm2, %xmm3

// CHECK: vpandn  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xdf,0xd9]
          vpandn  %xmm1, %xmm2, %xmm3

// CHECK: vpandn  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xdf,0x18]
          vpandn  (%eax), %xmm2, %xmm3

// CHECK: vpcmpeqb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x74,0xd9]
          vpcmpeqb  %xmm1, %xmm2, %xmm3

// CHECK: vpcmpeqb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x74,0x18]
          vpcmpeqb  (%eax), %xmm2, %xmm3

// CHECK: vpcmpeqw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x75,0xd9]
          vpcmpeqw  %xmm1, %xmm2, %xmm3

// CHECK: vpcmpeqw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x75,0x18]
          vpcmpeqw  (%eax), %xmm2, %xmm3

// CHECK: vpcmpeqd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x76,0xd9]
          vpcmpeqd  %xmm1, %xmm2, %xmm3

// CHECK: vpcmpeqd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x76,0x18]
          vpcmpeqd  (%eax), %xmm2, %xmm3

// CHECK: vpcmpgtb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x64,0xd9]
          vpcmpgtb  %xmm1, %xmm2, %xmm3

// CHECK: vpcmpgtb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x64,0x18]
          vpcmpgtb  (%eax), %xmm2, %xmm3

// CHECK: vpcmpgtw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x65,0xd9]
          vpcmpgtw  %xmm1, %xmm2, %xmm3

// CHECK: vpcmpgtw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x65,0x18]
          vpcmpgtw  (%eax), %xmm2, %xmm3

// CHECK: vpcmpgtd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x66,0xd9]
          vpcmpgtd  %xmm1, %xmm2, %xmm3

// CHECK: vpcmpgtd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x66,0x18]
          vpcmpgtd  (%eax), %xmm2, %xmm3

// CHECK: vpacksswb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x63,0xd9]
          vpacksswb  %xmm1, %xmm2, %xmm3

// CHECK: vpacksswb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x63,0x18]
          vpacksswb  (%eax), %xmm2, %xmm3

// CHECK: vpackssdw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x6b,0xd9]
          vpackssdw  %xmm1, %xmm2, %xmm3

// CHECK: vpackssdw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x6b,0x18]
          vpackssdw  (%eax), %xmm2, %xmm3

// CHECK: vpackuswb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x67,0xd9]
          vpackuswb  %xmm1, %xmm2, %xmm3

// CHECK: vpackuswb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x67,0x18]
          vpackuswb  (%eax), %xmm2, %xmm3

// CHECK: vpshufd  $4, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xf9,0x70,0xda,0x04]
          vpshufd  $4, %xmm2, %xmm3

// CHECK: vpshufd  $4, (%eax), %xmm3
// CHECK: encoding: [0xc5,0xf9,0x70,0x18,0x04]
          vpshufd  $4, (%eax), %xmm3

// CHECK: vpshufhw  $4, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xfa,0x70,0xda,0x04]
          vpshufhw  $4, %xmm2, %xmm3

// CHECK: vpshufhw  $4, (%eax), %xmm3
// CHECK: encoding: [0xc5,0xfa,0x70,0x18,0x04]
          vpshufhw  $4, (%eax), %xmm3

// CHECK: vpshuflw  $4, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xfb,0x70,0xda,0x04]
          vpshuflw  $4, %xmm2, %xmm3

// CHECK: vpshuflw  $4, (%eax), %xmm3
// CHECK: encoding: [0xc5,0xfb,0x70,0x18,0x04]
          vpshuflw  $4, (%eax), %xmm3

// CHECK: vpunpcklbw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x60,0xd9]
          vpunpcklbw  %xmm1, %xmm2, %xmm3

// CHECK: vpunpcklbw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x60,0x18]
          vpunpcklbw  (%eax), %xmm2, %xmm3

// CHECK: vpunpcklwd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x61,0xd9]
          vpunpcklwd  %xmm1, %xmm2, %xmm3

// CHECK: vpunpcklwd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x61,0x18]
          vpunpcklwd  (%eax), %xmm2, %xmm3

// CHECK: vpunpckldq  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x62,0xd9]
          vpunpckldq  %xmm1, %xmm2, %xmm3

// CHECK: vpunpckldq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x62,0x18]
          vpunpckldq  (%eax), %xmm2, %xmm3

// CHECK: vpunpcklqdq  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x6c,0xd9]
          vpunpcklqdq  %xmm1, %xmm2, %xmm3

// CHECK: vpunpcklqdq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x6c,0x18]
          vpunpcklqdq  (%eax), %xmm2, %xmm3

// CHECK: vpunpckhbw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x68,0xd9]
          vpunpckhbw  %xmm1, %xmm2, %xmm3

// CHECK: vpunpckhbw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x68,0x18]
          vpunpckhbw  (%eax), %xmm2, %xmm3

// CHECK: vpunpckhwd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x69,0xd9]
          vpunpckhwd  %xmm1, %xmm2, %xmm3

// CHECK: vpunpckhwd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x69,0x18]
          vpunpckhwd  (%eax), %xmm2, %xmm3

// CHECK: vpunpckhdq  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x6a,0xd9]
          vpunpckhdq  %xmm1, %xmm2, %xmm3

// CHECK: vpunpckhdq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x6a,0x18]
          vpunpckhdq  (%eax), %xmm2, %xmm3

// CHECK: vpunpckhqdq  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x6d,0xd9]
          vpunpckhqdq  %xmm1, %xmm2, %xmm3

// CHECK: vpunpckhqdq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x6d,0x18]
          vpunpckhqdq  (%eax), %xmm2, %xmm3

// CHECK: vpinsrw  $7, %eax, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc4,0xd8,0x07]
          vpinsrw  $7, %eax, %xmm2, %xmm3

// CHECK: vpinsrw  $7, (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xc4,0x18,0x07]
          vpinsrw  $7, (%eax), %xmm2, %xmm3

// CHECK: vpextrw  $7, %xmm2, %eax
// CHECK: encoding: [0xc5,0xf9,0xc5,0xc2,0x07]
          vpextrw  $7, %xmm2, %eax

// CHECK: vpmovmskb  %xmm1, %eax
// CHECK: encoding: [0xc5,0xf9,0xd7,0xc1]
          vpmovmskb  %xmm1, %eax

// CHECK: vmaskmovdqu  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf9,0xf7,0xd1]
          vmaskmovdqu  %xmm1, %xmm2

// CHECK: vmovd  %xmm1, %eax
// CHECK: encoding: [0xc5,0xf9,0x7e,0xc8]
          vmovd  %xmm1, %eax

// CHECK: vmovd  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf9,0x7e,0x08]
          vmovd  %xmm1, (%eax)

// CHECK: vmovd  %eax, %xmm1
// CHECK: encoding: [0xc5,0xf9,0x6e,0xc8]
          vmovd  %eax, %xmm1

// CHECK: vmovd  (%eax), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x6e,0x08]
          vmovd  (%eax), %xmm1

// CHECK: vmovq  %xmm1, (%eax)
// CHECK: encoding: [0xc5,0xf9,0xd6,0x08]
          vmovq  %xmm1, (%eax)

// CHECK: vmovq  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xfa,0x7e,0xd1]
          vmovq  %xmm1, %xmm2

// CHECK: vmovq  (%eax), %xmm1
// CHECK: encoding: [0xc5,0xfa,0x7e,0x08]
          vmovq  (%eax), %xmm1

// CHECK: vcvtpd2dq  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xfb,0xe6,0xd1]
          vcvtpd2dq  %xmm1, %xmm2

// CHECK: vcvtdq2pd  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xfa,0xe6,0xd1]
          vcvtdq2pd  %xmm1, %xmm2

// CHECK: vcvtdq2pd  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xfa,0xe6,0x10]
          vcvtdq2pd  (%eax), %xmm2

// CHECK: vmovshdup  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xfa,0x16,0xd1]
          vmovshdup  %xmm1, %xmm2

// CHECK: vmovshdup  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xfa,0x16,0x10]
          vmovshdup  (%eax), %xmm2

// CHECK: vmovsldup  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xfa,0x12,0xd1]
          vmovsldup  %xmm1, %xmm2

// CHECK: vmovsldup  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xfa,0x12,0x10]
          vmovsldup  (%eax), %xmm2

// CHECK: vmovddup  %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xfb,0x12,0xd1]
          vmovddup  %xmm1, %xmm2

// CHECK: vmovddup  (%eax), %xmm2
// CHECK: encoding: [0xc5,0xfb,0x12,0x10]
          vmovddup  (%eax), %xmm2

// CHECK: vaddsubps  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0xd0,0xd9]
          vaddsubps  %xmm1, %xmm2, %xmm3

// CHECK: vaddsubps  (%eax), %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf3,0xd0,0x10]
          vaddsubps  (%eax), %xmm1, %xmm2

// CHECK: vaddsubpd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0xd0,0xd9]
          vaddsubpd  %xmm1, %xmm2, %xmm3

// CHECK: vaddsubpd  (%eax), %xmm1, %xmm2
// CHECK: encoding: [0xc5,0xf1,0xd0,0x10]
          vaddsubpd  (%eax), %xmm1, %xmm2

// CHECK: vhaddps  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0x7c,0xd9]
          vhaddps  %xmm1, %xmm2, %xmm3

// CHECK: vhaddps  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0x7c,0x18]
          vhaddps  (%eax), %xmm2, %xmm3

// CHECK: vhaddpd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x7c,0xd9]
          vhaddpd  %xmm1, %xmm2, %xmm3

// CHECK: vhaddpd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x7c,0x18]
          vhaddpd  (%eax), %xmm2, %xmm3

// CHECK: vhsubps  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0x7d,0xd9]
          vhsubps  %xmm1, %xmm2, %xmm3

// CHECK: vhsubps  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xeb,0x7d,0x18]
          vhsubps  (%eax), %xmm2, %xmm3

// CHECK: vhsubpd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x7d,0xd9]
          vhsubpd  %xmm1, %xmm2, %xmm3

// CHECK: vhsubpd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe9,0x7d,0x18]
          vhsubpd  (%eax), %xmm2, %xmm3

// CHECK: vpabsb  %xmm1, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0xd1]
          vpabsb  %xmm1, %xmm2

// CHECK: vpabsb  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x1c,0x10]
          vpabsb  (%eax), %xmm2

// CHECK: vpabsw  %xmm1, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0xd1]
          vpabsw  %xmm1, %xmm2

// CHECK: vpabsw  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x1d,0x10]
          vpabsw  (%eax), %xmm2

// CHECK: vpabsd  %xmm1, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0xd1]
          vpabsd  %xmm1, %xmm2

// CHECK: vpabsd  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x1e,0x10]
          vpabsd  (%eax), %xmm2

// CHECK: vphaddw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x01,0xd9]
          vphaddw  %xmm1, %xmm2, %xmm3

// CHECK: vphaddw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x01,0x18]
          vphaddw  (%eax), %xmm2, %xmm3

// CHECK: vphaddd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x02,0xd9]
          vphaddd  %xmm1, %xmm2, %xmm3

// CHECK: vphaddd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x02,0x18]
          vphaddd  (%eax), %xmm2, %xmm3

// CHECK: vphaddsw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x03,0xd9]
          vphaddsw  %xmm1, %xmm2, %xmm3

// CHECK: vphaddsw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x03,0x18]
          vphaddsw  (%eax), %xmm2, %xmm3

// CHECK: vphsubw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x05,0xd9]
          vphsubw  %xmm1, %xmm2, %xmm3

// CHECK: vphsubw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x05,0x18]
          vphsubw  (%eax), %xmm2, %xmm3

// CHECK: vphsubd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x06,0xd9]
          vphsubd  %xmm1, %xmm2, %xmm3

// CHECK: vphsubd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x06,0x18]
          vphsubd  (%eax), %xmm2, %xmm3

// CHECK: vphsubsw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x07,0xd9]
          vphsubsw  %xmm1, %xmm2, %xmm3

// CHECK: vphsubsw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x07,0x18]
          vphsubsw  (%eax), %xmm2, %xmm3

// CHECK: vpmaddubsw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x04,0xd9]
          vpmaddubsw  %xmm1, %xmm2, %xmm3

// CHECK: vpmaddubsw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x04,0x18]
          vpmaddubsw  (%eax), %xmm2, %xmm3

// CHECK: vpshufb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x00,0xd9]
          vpshufb  %xmm1, %xmm2, %xmm3

// CHECK: vpshufb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x00,0x18]
          vpshufb  (%eax), %xmm2, %xmm3

// CHECK: vpsignb  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x08,0xd9]
          vpsignb  %xmm1, %xmm2, %xmm3

// CHECK: vpsignb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x08,0x18]
          vpsignb  (%eax), %xmm2, %xmm3

// CHECK: vpsignw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x09,0xd9]
          vpsignw  %xmm1, %xmm2, %xmm3

// CHECK: vpsignw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x09,0x18]
          vpsignw  (%eax), %xmm2, %xmm3

// CHECK: vpsignd  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x0a,0xd9]
          vpsignd  %xmm1, %xmm2, %xmm3

// CHECK: vpsignd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x0a,0x18]
          vpsignd  (%eax), %xmm2, %xmm3

// CHECK: vpmulhrsw  %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x0b,0xd9]
          vpmulhrsw  %xmm1, %xmm2, %xmm3

// CHECK: vpmulhrsw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x0b,0x18]
          vpmulhrsw  (%eax), %xmm2, %xmm3

// CHECK: vpalignr  $7, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x69,0x0f,0xd9,0x07]
          vpalignr  $7, %xmm1, %xmm2, %xmm3

// CHECK: vpalignr  $7, (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x69,0x0f,0x18,0x07]
          vpalignr  $7, (%eax), %xmm2, %xmm3

// CHECK: vroundsd  $7, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x69,0x0b,0xd9,0x07]
          vroundsd  $7, %xmm1, %xmm2, %xmm3

// CHECK: vroundsd  $7, (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x69,0x0b,0x18,0x07]
          vroundsd  $7, (%eax), %xmm2, %xmm3

// CHECK: vroundss  $7, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x69,0x0a,0xd9,0x07]
          vroundss  $7, %xmm1, %xmm2, %xmm3

// CHECK: vroundss  $7, (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x69,0x0a,0x18,0x07]
          vroundss  $7, (%eax), %xmm2, %xmm3

// CHECK: vroundpd  $7, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0xda,0x07]
          vroundpd  $7, %xmm2, %xmm3

// CHECK: vroundpd  $7, (%eax), %xmm3
// CHECK: encoding: [0xc4,0xe3,0x79,0x09,0x18,0x07]
          vroundpd  $7, (%eax), %xmm3

// CHECK: vroundps  $7, %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0xda,0x07]
          vroundps  $7, %xmm2, %xmm3

// CHECK: vroundps  $7, (%eax), %xmm3
// CHECK: encoding: [0xc4,0xe3,0x79,0x08,0x18,0x07]
          vroundps  $7, (%eax), %xmm3

// CHECK: vphminposuw  %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0xda]
          vphminposuw  %xmm2, %xmm3

// CHECK: vphminposuw  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x41,0x10]
          vphminposuw  (%eax), %xmm2

// CHECK: vpackusdw  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x2b,0xca]
          vpackusdw  %xmm2, %xmm3, %xmm1

// CHECK: vpackusdw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x2b,0x18]
          vpackusdw  (%eax), %xmm2, %xmm3

// CHECK: vpcmpeqq  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x29,0xca]
          vpcmpeqq  %xmm2, %xmm3, %xmm1

// CHECK: vpcmpeqq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x29,0x18]
          vpcmpeqq  (%eax), %xmm2, %xmm3

// CHECK: vpminsb  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x38,0xca]
          vpminsb  %xmm2, %xmm3, %xmm1

// CHECK: vpminsb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x38,0x18]
          vpminsb  (%eax), %xmm2, %xmm3

// CHECK: vpminsd  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x39,0xca]
          vpminsd  %xmm2, %xmm3, %xmm1

// CHECK: vpminsd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x39,0x18]
          vpminsd  (%eax), %xmm2, %xmm3

// CHECK: vpminud  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x3b,0xca]
          vpminud  %xmm2, %xmm3, %xmm1

// CHECK: vpminud  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x3b,0x18]
          vpminud  (%eax), %xmm2, %xmm3

// CHECK: vpminuw  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x3a,0xca]
          vpminuw  %xmm2, %xmm3, %xmm1

// CHECK: vpminuw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x3a,0x18]
          vpminuw  (%eax), %xmm2, %xmm3

// CHECK: vpmaxsb  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x3c,0xca]
          vpmaxsb  %xmm2, %xmm3, %xmm1

// CHECK: vpmaxsb  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x3c,0x18]
          vpmaxsb  (%eax), %xmm2, %xmm3

// CHECK: vpmaxsd  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x3d,0xca]
          vpmaxsd  %xmm2, %xmm3, %xmm1

// CHECK: vpmaxsd  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x3d,0x18]
          vpmaxsd  (%eax), %xmm2, %xmm3

// CHECK: vpmaxud  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x3f,0xca]
          vpmaxud  %xmm2, %xmm3, %xmm1

// CHECK: vpmaxud  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x3f,0x18]
          vpmaxud  (%eax), %xmm2, %xmm3

// CHECK: vpmaxuw  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x3e,0xca]
          vpmaxuw  %xmm2, %xmm3, %xmm1

// CHECK: vpmaxuw  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x3e,0x18]
          vpmaxuw  (%eax), %xmm2, %xmm3

// CHECK: vpmuldq  %xmm2, %xmm3, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x61,0x28,0xca]
          vpmuldq  %xmm2, %xmm3, %xmm1

// CHECK: vpmuldq  (%eax), %xmm2, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x69,0x28,0x18]
          vpmuldq  (%eax), %xmm2, %xmm3

// CHECK: vpmulld  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x40,0xca]
          vpmulld  %xmm2, %xmm5, %xmm1

// CHECK: vpmulld  (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x51,0x40,0x18]
          vpmulld  (%eax), %xmm5, %xmm3

// CHECK: vblendps  $3, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x0c,0xca,0x03]
          vblendps  $3, %xmm2, %xmm5, %xmm1

// CHECK: vblendps  $3, (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x0c,0x08,0x03]
          vblendps  $3, (%eax), %xmm5, %xmm1

// CHECK: vblendpd  $3, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x0d,0xca,0x03]
          vblendpd  $3, %xmm2, %xmm5, %xmm1

// CHECK: vblendpd  $3, (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x0d,0x08,0x03]
          vblendpd  $3, (%eax), %xmm5, %xmm1

// CHECK: vpblendw  $3, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x0e,0xca,0x03]
          vpblendw  $3, %xmm2, %xmm5, %xmm1

// CHECK: vpblendw  $3, (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x0e,0x08,0x03]
          vpblendw  $3, (%eax), %xmm5, %xmm1

// CHECK: vmpsadbw  $3, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x42,0xca,0x03]
          vmpsadbw  $3, %xmm2, %xmm5, %xmm1

// CHECK: vmpsadbw  $3, (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x42,0x08,0x03]
          vmpsadbw  $3, (%eax), %xmm5, %xmm1

// CHECK: vdpps  $3, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x40,0xca,0x03]
          vdpps  $3, %xmm2, %xmm5, %xmm1

// CHECK: vdpps  $3, (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x40,0x08,0x03]
          vdpps  $3, (%eax), %xmm5, %xmm1

// CHECK: vdppd  $3, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x41,0xca,0x03]
          vdppd  $3, %xmm2, %xmm5, %xmm1

// CHECK: vdppd  $3, (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x41,0x08,0x03]
          vdppd  $3, (%eax), %xmm5, %xmm1

// CHECK: vblendvpd  %xmm2, %xmm5, %xmm1, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x71,0x4b,0xdd,0x20]
          vblendvpd  %xmm2, %xmm5, %xmm1, %xmm3

// CHECK: vblendvpd  %xmm2, (%eax), %xmm1, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x71,0x4b,0x18,0x20]
          vblendvpd  %xmm2, (%eax), %xmm1, %xmm3

// CHECK: vblendvps  %xmm2, %xmm5, %xmm1, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x71,0x4a,0xdd,0x20]
          vblendvps  %xmm2, %xmm5, %xmm1, %xmm3

// CHECK: vblendvps  %xmm2, (%eax), %xmm1, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x71,0x4a,0x18,0x20]
          vblendvps  %xmm2, (%eax), %xmm1, %xmm3

// CHECK: vpblendvb  %xmm2, %xmm5, %xmm1, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x71,0x4c,0xdd,0x20]
          vpblendvb  %xmm2, %xmm5, %xmm1, %xmm3

// CHECK: vpblendvb  %xmm2, (%eax), %xmm1, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x71,0x4c,0x18,0x20]
          vpblendvb  %xmm2, (%eax), %xmm1, %xmm3

// CHECK: vpmovsxbw  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0xea]
          vpmovsxbw  %xmm2, %xmm5

// CHECK: vpmovsxbw  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x20,0x10]
          vpmovsxbw  (%eax), %xmm2

// CHECK: vpmovsxwd  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0xea]
          vpmovsxwd  %xmm2, %xmm5

// CHECK: vpmovsxwd  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x23,0x10]
          vpmovsxwd  (%eax), %xmm2

// CHECK: vpmovsxdq  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0xea]
          vpmovsxdq  %xmm2, %xmm5

// CHECK: vpmovsxdq  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x25,0x10]
          vpmovsxdq  (%eax), %xmm2

// CHECK: vpmovzxbw  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0xea]
          vpmovzxbw  %xmm2, %xmm5

// CHECK: vpmovzxbw  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x30,0x10]
          vpmovzxbw  (%eax), %xmm2

// CHECK: vpmovzxwd  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0xea]
          vpmovzxwd  %xmm2, %xmm5

// CHECK: vpmovzxwd  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x33,0x10]
          vpmovzxwd  (%eax), %xmm2

// CHECK: vpmovzxdq  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0xea]
          vpmovzxdq  %xmm2, %xmm5

// CHECK: vpmovzxdq  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x35,0x10]
          vpmovzxdq  (%eax), %xmm2

// CHECK: vpmovsxbq  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0xea]
          vpmovsxbq  %xmm2, %xmm5

// CHECK: vpmovsxbq  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x22,0x10]
          vpmovsxbq  (%eax), %xmm2

// CHECK: vpmovzxbq  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0xea]
          vpmovzxbq  %xmm2, %xmm5

// CHECK: vpmovzxbq  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x32,0x10]
          vpmovzxbq  (%eax), %xmm2

// CHECK: vpmovsxbd  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0xea]
          vpmovsxbd  %xmm2, %xmm5

// CHECK: vpmovsxbd  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x21,0x10]
          vpmovsxbd  (%eax), %xmm2

// CHECK: vpmovsxwq  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0xea]
          vpmovsxwq  %xmm2, %xmm5

// CHECK: vpmovsxwq  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x24,0x10]
          vpmovsxwq  (%eax), %xmm2

// CHECK: vpmovzxbd  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0xea]
          vpmovzxbd  %xmm2, %xmm5

// CHECK: vpmovzxbd  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x31,0x10]
          vpmovzxbd  (%eax), %xmm2

// CHECK: vpmovzxwq  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0xea]
          vpmovzxwq  %xmm2, %xmm5

// CHECK: vpmovzxwq  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x34,0x10]
          vpmovzxwq  (%eax), %xmm2

// CHECK: vpextrw  $7, %xmm2, %eax
// CHECK: encoding: [0xc5,0xf9,0xc5,0xc2,0x07]
          vpextrw  $7, %xmm2, %eax

// CHECK: vpextrw  $7, %xmm2, (%eax)
// CHECK: encoding: [0xc4,0xe3,0x79,0x15,0x10,0x07]
          vpextrw  $7, %xmm2, (%eax)

// CHECK: vpextrd  $7, %xmm2, %eax
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0xd0,0x07]
          vpextrd  $7, %xmm2, %eax

// CHECK: vpextrd  $7, %xmm2, (%eax)
// CHECK: encoding: [0xc4,0xe3,0x79,0x16,0x10,0x07]
          vpextrd  $7, %xmm2, (%eax)

// CHECK: vpextrb  $7, %xmm2, %eax
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0xd0,0x07]
          vpextrb  $7, %xmm2, %eax

// CHECK: vpextrb  $7, %xmm2, (%eax)
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0x10,0x07]
          vpextrb  $7, %xmm2, (%eax)

// CHECK: vextractps  $7, %xmm2, (%eax)
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0x10,0x07]
          vextractps  $7, %xmm2, (%eax)

// CHECK: vextractps  $7, %xmm2, %eax
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0xd0,0x07]
          vextractps  $7, %xmm2, %eax

// CHECK: vpinsrw  $7, %eax, %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0xc4,0xe8,0x07]
          vpinsrw  $7, %eax, %xmm2, %xmm5

// CHECK: vpinsrw  $7, (%eax), %xmm2, %xmm5
// CHECK: encoding: [0xc5,0xe9,0xc4,0x28,0x07]
          vpinsrw  $7, (%eax), %xmm2, %xmm5

// CHECK: vpinsrb  $7, %eax, %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x69,0x20,0xe8,0x07]
          vpinsrb  $7, %eax, %xmm2, %xmm5

// CHECK: vpinsrb  $7, (%eax), %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x69,0x20,0x28,0x07]
          vpinsrb  $7, (%eax), %xmm2, %xmm5

// CHECK: vpinsrd  $7, %eax, %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x69,0x22,0xe8,0x07]
          vpinsrd  $7, %eax, %xmm2, %xmm5

// CHECK: vpinsrd  $7, (%eax), %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x69,0x22,0x28,0x07]
          vpinsrd  $7, (%eax), %xmm2, %xmm5

// CHECK: vinsertps  $7, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x21,0xca,0x07]
          vinsertps  $7, %xmm2, %xmm5, %xmm1

// CHECK: vinsertps  $7, (%eax), %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x21,0x08,0x07]
          vinsertps  $7, (%eax), %xmm5, %xmm1

// CHECK: vptest  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0xea]
          vptest  %xmm2, %xmm5

// CHECK: vptest  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x17,0x10]
          vptest  (%eax), %xmm2

// CHECK: vmovntdqa  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x2a,0x10]
          vmovntdqa  (%eax), %xmm2

// CHECK: vpcmpgtq  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x37,0xca]
          vpcmpgtq  %xmm2, %xmm5, %xmm1

// CHECK: vpcmpgtq  (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x51,0x37,0x18]
          vpcmpgtq  (%eax), %xmm5, %xmm3

// CHECK: vpcmpistrm  $7, %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0xea,0x07]
          vpcmpistrm  $7, %xmm2, %xmm5

// CHECK: vpcmpistrm  $7, (%eax), %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x62,0x28,0x07]
          vpcmpistrm  $7, (%eax), %xmm5

// CHECK: vpcmpestrm  $7, %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0xea,0x07]
          vpcmpestrm  $7, %xmm2, %xmm5

// CHECK: vpcmpestrm  $7, (%eax), %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x60,0x28,0x07]
          vpcmpestrm  $7, (%eax), %xmm5

// CHECK: vpcmpistri  $7, %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0xea,0x07]
          vpcmpistri  $7, %xmm2, %xmm5

// CHECK: vpcmpistri  $7, (%eax), %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x63,0x28,0x07]
          vpcmpistri  $7, (%eax), %xmm5

// CHECK: vpcmpestri  $7, %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0xea,0x07]
          vpcmpestri  $7, %xmm2, %xmm5

// CHECK: vpcmpestri  $7, (%eax), %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x61,0x28,0x07]
          vpcmpestri  $7, (%eax), %xmm5

// CHECK: vaesimc  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0xdb,0xea]
          vaesimc  %xmm2, %xmm5

// CHECK: vaesimc  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0xdb,0x10]
          vaesimc  (%eax), %xmm2

// CHECK: vaesenc  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xdc,0xca]
          vaesenc  %xmm2, %xmm5, %xmm1

// CHECK: vaesenc  (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x51,0xdc,0x18]
          vaesenc  (%eax), %xmm5, %xmm3

// CHECK: vaesenclast  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xdd,0xca]
          vaesenclast  %xmm2, %xmm5, %xmm1

// CHECK: vaesenclast  (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x51,0xdd,0x18]
          vaesenclast  (%eax), %xmm5, %xmm3

// CHECK: vaesdec  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xde,0xca]
          vaesdec  %xmm2, %xmm5, %xmm1

// CHECK: vaesdec  (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x51,0xde,0x18]
          vaesdec  (%eax), %xmm5, %xmm3

// CHECK: vaesdeclast  %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0xdf,0xca]
          vaesdeclast  %xmm2, %xmm5, %xmm1

// CHECK: vaesdeclast  (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x51,0xdf,0x18]
          vaesdeclast  (%eax), %xmm5, %xmm3

// CHECK: vaeskeygenassist  $7, %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0xdf,0xea,0x07]
          vaeskeygenassist  $7, %xmm2, %xmm5

// CHECK: vaeskeygenassist  $7, (%eax), %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0xdf,0x28,0x07]
          vaeskeygenassist  $7, (%eax), %xmm5

// CHECK: vcmpps  $8, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x08]
          vcmpeq_uqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $9, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x09]
          vcmpngeps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $10, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x0a]
          vcmpngtps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $11, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x0b]
          vcmpfalseps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $12, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x0c]
          vcmpneq_oqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $13, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x0d]
          vcmpgeps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $14, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x0e]
          vcmpgtps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $15, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x0f]
          vcmptrueps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $16, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x10]
          vcmpeq_osps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $17, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x11]
          vcmplt_oqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $18, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x12]
          vcmple_oqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $19, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x13]
          vcmpunord_sps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $20, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x14]
          vcmpneq_usps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $21, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x15]
          vcmpnlt_uqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $22, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x16]
          vcmpnle_uqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $23, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x17]
          vcmpord_sps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $24, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x18]
          vcmpeq_usps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $25, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x19]
          vcmpnge_uqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $26, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x1a]
          vcmpngt_uqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $27, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x1b]
          vcmpfalse_osps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $28, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x1c]
          vcmpneq_osps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $29, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x1d]
          vcmpge_oqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $30, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x1e]
          vcmpgt_oqps %xmm1, %xmm2, %xmm3

// CHECK: vcmpps  $31, %xmm1, %xmm2, %xmm3
// CHECK: encoding: [0xc5,0xe8,0xc2,0xd9,0x1f]
          vcmptrue_usps %xmm1, %xmm2, %xmm3

// CHECK: vmovaps  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfc,0x28,0x10]
          vmovaps  (%eax), %ymm2

// CHECK: vmovaps  %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xfc,0x28,0xd1]
          vmovaps  %ymm1, %ymm2

// CHECK: vmovaps  %ymm1, (%eax)
// CHECK: encoding: [0xc5,0xfc,0x29,0x08]
          vmovaps  %ymm1, (%eax)

// CHECK: vmovapd  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfd,0x28,0x10]
          vmovapd  (%eax), %ymm2

// CHECK: vmovapd  %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xfd,0x28,0xd1]
          vmovapd  %ymm1, %ymm2

// CHECK: vmovapd  %ymm1, (%eax)
// CHECK: encoding: [0xc5,0xfd,0x29,0x08]
          vmovapd  %ymm1, (%eax)

// CHECK: vmovups  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfc,0x10,0x10]
          vmovups  (%eax), %ymm2

// CHECK: vmovups  %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xfc,0x10,0xd1]
          vmovups  %ymm1, %ymm2

// CHECK: vmovups  %ymm1, (%eax)
// CHECK: encoding: [0xc5,0xfc,0x11,0x08]
          vmovups  %ymm1, (%eax)

// CHECK: vmovupd  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfd,0x10,0x10]
          vmovupd  (%eax), %ymm2

// CHECK: vmovupd  %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xfd,0x10,0xd1]
          vmovupd  %ymm1, %ymm2

// CHECK: vmovupd  %ymm1, (%eax)
// CHECK: encoding: [0xc5,0xfd,0x11,0x08]
          vmovupd  %ymm1, (%eax)

// CHECK: vunpckhps  %ymm1, %ymm2, %ymm4
// CHECK: encoding: [0xc5,0xec,0x15,0xe1]
          vunpckhps  %ymm1, %ymm2, %ymm4

// CHECK: vunpckhpd  %ymm1, %ymm2, %ymm4
// CHECK: encoding: [0xc5,0xed,0x15,0xe1]
          vunpckhpd  %ymm1, %ymm2, %ymm4

// CHECK: vunpcklps  %ymm1, %ymm2, %ymm4
// CHECK: encoding: [0xc5,0xec,0x14,0xe1]
          vunpcklps  %ymm1, %ymm2, %ymm4

// CHECK: vunpcklpd  %ymm1, %ymm2, %ymm4
// CHECK: encoding: [0xc5,0xed,0x14,0xe1]
          vunpcklpd  %ymm1, %ymm2, %ymm4

// CHECK: vunpckhps  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xec,0x15,0x6c,0xcb,0xfc]
          vunpckhps  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vunpckhpd  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xed,0x15,0x6c,0xcb,0xfc]
          vunpckhpd  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vunpcklps  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xec,0x14,0x6c,0xcb,0xfc]
          vunpcklps  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vunpcklpd  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xed,0x14,0x6c,0xcb,0xfc]
          vunpcklpd  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vmovntdq  %ymm1, (%eax)
// CHECK: encoding: [0xc5,0xfd,0xe7,0x08]
          vmovntdq  %ymm1, (%eax)

// CHECK: vmovntpd  %ymm1, (%eax)
// CHECK: encoding: [0xc5,0xfd,0x2b,0x08]
          vmovntpd  %ymm1, (%eax)

// CHECK: vmovntps  %ymm1, (%eax)
// CHECK: encoding: [0xc5,0xfc,0x2b,0x08]
          vmovntps  %ymm1, (%eax)

// CHECK: vmovmskps  %xmm2, %eax
// CHECK: encoding: [0xc5,0xf8,0x50,0xc2]
          vmovmskps  %xmm2, %eax

// CHECK: vmovmskpd  %xmm2, %eax
// CHECK: encoding: [0xc5,0xf9,0x50,0xc2]
          vmovmskpd  %xmm2, %eax

// CHECK: vmaxps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5f,0xf2]
          vmaxps  %ymm2, %ymm4, %ymm6

// CHECK: vmaxpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5f,0xf2]
          vmaxpd  %ymm2, %ymm4, %ymm6

// CHECK: vminps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5d,0xf2]
          vminps  %ymm2, %ymm4, %ymm6

// CHECK: vminpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5d,0xf2]
          vminpd  %ymm2, %ymm4, %ymm6

// CHECK: vsubps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5c,0xf2]
          vsubps  %ymm2, %ymm4, %ymm6

// CHECK: vsubpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5c,0xf2]
          vsubpd  %ymm2, %ymm4, %ymm6

// CHECK: vdivps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5e,0xf2]
          vdivps  %ymm2, %ymm4, %ymm6

// CHECK: vdivpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5e,0xf2]
          vdivpd  %ymm2, %ymm4, %ymm6

// CHECK: vaddps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x58,0xf2]
          vaddps  %ymm2, %ymm4, %ymm6

// CHECK: vaddpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x58,0xf2]
          vaddpd  %ymm2, %ymm4, %ymm6

// CHECK: vmulps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x59,0xf2]
          vmulps  %ymm2, %ymm4, %ymm6

// CHECK: vmulpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x59,0xf2]
          vmulpd  %ymm2, %ymm4, %ymm6

// CHECK: vmaxps  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5f,0x30]
          vmaxps  (%eax), %ymm4, %ymm6

// CHECK: vmaxpd  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5f,0x30]
          vmaxpd  (%eax), %ymm4, %ymm6

// CHECK: vminps  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5d,0x30]
          vminps  (%eax), %ymm4, %ymm6

// CHECK: vminpd  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5d,0x30]
          vminpd  (%eax), %ymm4, %ymm6

// CHECK: vsubps  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5c,0x30]
          vsubps  (%eax), %ymm4, %ymm6

// CHECK: vsubpd  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5c,0x30]
          vsubpd  (%eax), %ymm4, %ymm6

// CHECK: vdivps  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5e,0x30]
          vdivps  (%eax), %ymm4, %ymm6

// CHECK: vdivpd  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5e,0x30]
          vdivpd  (%eax), %ymm4, %ymm6

// CHECK: vaddps  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x58,0x30]
          vaddps  (%eax), %ymm4, %ymm6

// CHECK: vaddpd  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x58,0x30]
          vaddpd  (%eax), %ymm4, %ymm6

// CHECK: vmulps  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x59,0x30]
          vmulps  (%eax), %ymm4, %ymm6

// CHECK: vmulpd  (%eax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x59,0x30]
          vmulpd  (%eax), %ymm4, %ymm6

// CHECK: vsqrtpd  %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xfd,0x51,0xd1]
          vsqrtpd  %ymm1, %ymm2

// CHECK: vsqrtpd  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfd,0x51,0x10]
          vsqrtpd  (%eax), %ymm2

// CHECK: vsqrtps  %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xfc,0x51,0xd1]
          vsqrtps  %ymm1, %ymm2

// CHECK: vsqrtps  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfc,0x51,0x10]
          vsqrtps  (%eax), %ymm2

// CHECK: vrsqrtps  %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xfc,0x52,0xd1]
          vrsqrtps  %ymm1, %ymm2

// CHECK: vrsqrtps  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfc,0x52,0x10]
          vrsqrtps  (%eax), %ymm2

// CHECK: vrcpps  %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xfc,0x53,0xd1]
          vrcpps  %ymm1, %ymm2

// CHECK: vrcpps  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfc,0x53,0x10]
          vrcpps  (%eax), %ymm2

// CHECK: vandps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x54,0xf2]
          vandps  %ymm2, %ymm4, %ymm6

// CHECK: vandpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x54,0xf2]
          vandpd  %ymm2, %ymm4, %ymm6

// CHECK: vandps  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xec,0x54,0x6c,0xcb,0xfc]
          vandps  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vandpd  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xed,0x54,0x6c,0xcb,0xfc]
          vandpd  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vorps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x56,0xf2]
          vorps  %ymm2, %ymm4, %ymm6

// CHECK: vorpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x56,0xf2]
          vorpd  %ymm2, %ymm4, %ymm6

// CHECK: vorps  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xec,0x56,0x6c,0xcb,0xfc]
          vorps  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vorpd  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xed,0x56,0x6c,0xcb,0xfc]
          vorpd  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vxorps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x57,0xf2]
          vxorps  %ymm2, %ymm4, %ymm6

// CHECK: vxorpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x57,0xf2]
          vxorpd  %ymm2, %ymm4, %ymm6

// CHECK: vxorps  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xec,0x57,0x6c,0xcb,0xfc]
          vxorps  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vxorpd  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xed,0x57,0x6c,0xcb,0xfc]
          vxorpd  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vandnps  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x55,0xf2]
          vandnps  %ymm2, %ymm4, %ymm6

// CHECK: vandnpd  %ymm2, %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x55,0xf2]
          vandnpd  %ymm2, %ymm4, %ymm6

// CHECK: vandnps  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xec,0x55,0x6c,0xcb,0xfc]
          vandnps  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vandnpd  -4(%ebx,%ecx,8), %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xed,0x55,0x6c,0xcb,0xfc]
          vandnpd  -4(%ebx,%ecx,8), %ymm2, %ymm5

// CHECK: vcvtps2pd  %xmm3, %ymm2
// CHECK: encoding: [0xc5,0xfc,0x5a,0xd3]
          vcvtps2pd  %xmm3, %ymm2

// CHECK: vcvtps2pd  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfc,0x5a,0x10]
          vcvtps2pd  (%eax), %ymm2

// CHECK: vcvtdq2pd  %xmm3, %ymm2
// CHECK: encoding: [0xc5,0xfe,0xe6,0xd3]
          vcvtdq2pd  %xmm3, %ymm2

// CHECK: vcvtdq2pd  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfe,0xe6,0x10]
          vcvtdq2pd  (%eax), %ymm2

// CHECK: vcvtdq2ps  %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xfc,0x5b,0xea]
          vcvtdq2ps  %ymm2, %ymm5

// CHECK: vcvtdq2ps  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfc,0x5b,0x10]
          vcvtdq2ps  (%eax), %ymm2

// CHECK: vcvtps2dq  %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xfd,0x5b,0xea]
          vcvtps2dq  %ymm2, %ymm5

// CHECK: vcvtps2dq  (%eax), %ymm5
// CHECK: encoding: [0xc5,0xfd,0x5b,0x28]
          vcvtps2dq  (%eax), %ymm5

// CHECK: vcvttps2dq  %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xfe,0x5b,0xea]
          vcvttps2dq  %ymm2, %ymm5

// CHECK: vcvttps2dq  (%eax), %ymm5
// CHECK: encoding: [0xc5,0xfe,0x5b,0x28]
          vcvttps2dq  (%eax), %ymm5

// CHECK: vcvttpd2dq  %xmm1, %xmm5
// CHECK: encoding: [0xc5,0xf9,0xe6,0xe9]
          vcvttpd2dq  %xmm1, %xmm5

// CHECK: vcvttpd2dqy %ymm2, %xmm5
// CHECK: encoding: [0xc5,0xfd,0xe6,0xea]
          vcvttpd2dq  %ymm2, %xmm5

// CHECK: vcvttpd2dq   %xmm1, %xmm5
// CHECK: encoding: [0xc5,0xf9,0xe6,0xe9]
          vcvttpd2dqx  %xmm1, %xmm5

// CHECK: vcvttpd2dqx  (%eax), %xmm1
// CHECK: encoding: [0xc5,0xf9,0xe6,0x08]
          vcvttpd2dqx  (%eax), %xmm1

// CHECK: vcvttpd2dqy  %ymm2, %xmm1
// CHECK: encoding: [0xc5,0xfd,0xe6,0xca]
          vcvttpd2dqy  %ymm2, %xmm1

// CHECK: vcvttpd2dqy  (%eax), %xmm1
// CHECK: encoding: [0xc5,0xfd,0xe6,0x08]
          vcvttpd2dqy  (%eax), %xmm1

// CHECK: vcvtpd2psy %ymm2, %xmm5
// CHECK: encoding: [0xc5,0xfd,0x5a,0xea]
          vcvtpd2ps  %ymm2, %xmm5

// CHECK: vcvtpd2ps   %xmm1, %xmm5
// CHECK: encoding: [0xc5,0xf9,0x5a,0xe9]
          vcvtpd2psx  %xmm1, %xmm5

// CHECK: vcvtpd2psx  (%eax), %xmm1
// CHECK: encoding: [0xc5,0xf9,0x5a,0x08]
          vcvtpd2psx  (%eax), %xmm1

// CHECK: vcvtpd2psy  %ymm2, %xmm1
// CHECK: encoding: [0xc5,0xfd,0x5a,0xca]
          vcvtpd2psy  %ymm2, %xmm1

// CHECK: vcvtpd2psy  (%eax), %xmm1
// CHECK: encoding: [0xc5,0xfd,0x5a,0x08]
          vcvtpd2psy  (%eax), %xmm1

// CHECK: vcvtpd2dqy %ymm2, %xmm5
// CHECK: encoding: [0xc5,0xff,0xe6,0xea]
          vcvtpd2dq  %ymm2, %xmm5

// CHECK: vcvtpd2dqy  %ymm2, %xmm1
// CHECK: encoding: [0xc5,0xff,0xe6,0xca]
          vcvtpd2dqy  %ymm2, %xmm1

// CHECK: vcvtpd2dqy  (%eax), %xmm1
// CHECK: encoding: [0xc5,0xff,0xe6,0x08]
          vcvtpd2dqy  (%eax), %xmm1

// CHECK: vcvtpd2dq   %xmm1, %xmm5
// CHECK: encoding: [0xc5,0xfb,0xe6,0xe9]
          vcvtpd2dqx  %xmm1, %xmm5

// CHECK: vcvtpd2dqx  (%eax), %xmm1
// CHECK: encoding: [0xc5,0xfb,0xe6,0x08]
          vcvtpd2dqx  (%eax), %xmm1

// CHECK: vcmpps  $0, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x00]
          vcmpeqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $2, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x02]
          vcmpleps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $1, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x01]
          vcmpltps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $4, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x04]
          vcmpneqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $6, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x06]
          vcmpnleps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $5, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x05]
          vcmpnltps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $7, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x07]
          vcmpordps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $3, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x03]
          vcmpunordps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $0, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0x5c,0xcb,0xfc,0x00]
          vcmpeqps -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmpps  $2, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0x5c,0xcb,0xfc,0x02]
          vcmpleps -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmpps  $1, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0x5c,0xcb,0xfc,0x01]
          vcmpltps -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmpps  $4, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0x5c,0xcb,0xfc,0x04]
          vcmpneqps -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmpps  $6, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0x5c,0xcb,0xfc,0x06]
          vcmpnleps -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmpps  $5, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0x5c,0xcb,0xfc,0x05]
          vcmpnltps -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmpps  $7, -4(%ebx,%ecx,8), %ymm6, %ymm2
// CHECK: encoding: [0xc5,0xcc,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordps -4(%ebx,%ecx,8), %ymm6, %ymm2

// CHECK: vcmpps  $3, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0x5c,0xcb,0xfc,0x03]
          vcmpunordps -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmppd  $0, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0xd9,0x00]
          vcmpeqpd %ymm1, %ymm2, %ymm3

// CHECK: vcmppd  $2, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0xd9,0x02]
          vcmplepd %ymm1, %ymm2, %ymm3

// CHECK: vcmppd  $1, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0xd9,0x01]
          vcmpltpd %ymm1, %ymm2, %ymm3

// CHECK: vcmppd  $4, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0xd9,0x04]
          vcmpneqpd %ymm1, %ymm2, %ymm3

// CHECK: vcmppd  $6, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0xd9,0x06]
          vcmpnlepd %ymm1, %ymm2, %ymm3

// CHECK: vcmppd  $5, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0xd9,0x05]
          vcmpnltpd %ymm1, %ymm2, %ymm3

// CHECK: vcmppd  $7, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0xd9,0x07]
          vcmpordpd %ymm1, %ymm2, %ymm3

// CHECK: vcmppd  $3, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0xd9,0x03]
          vcmpunordpd %ymm1, %ymm2, %ymm3

// CHECK: vcmppd  $0, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0x5c,0xcb,0xfc,0x00]
          vcmpeqpd -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmppd  $2, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0x5c,0xcb,0xfc,0x02]
          vcmplepd -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmppd  $1, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0x5c,0xcb,0xfc,0x01]
          vcmpltpd -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmppd  $4, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0x5c,0xcb,0xfc,0x04]
          vcmpneqpd -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmppd  $6, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0x5c,0xcb,0xfc,0x06]
          vcmpnlepd -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmppd  $5, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0x5c,0xcb,0xfc,0x05]
          vcmpnltpd -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmppd  $7, -4(%ebx,%ecx,8), %ymm6, %ymm2
// CHECK: encoding: [0xc5,0xcd,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordpd -4(%ebx,%ecx,8), %ymm6, %ymm2

// CHECK: vcmppd  $3, -4(%ebx,%ecx,8), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xc2,0x5c,0xcb,0xfc,0x03]
          vcmpunordpd -4(%ebx,%ecx,8), %ymm2, %ymm3

// CHECK: vcmpps  $8, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x08]
          vcmpeq_uqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $9, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x09]
          vcmpngeps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $10, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x0a]
          vcmpngtps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $11, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x0b]
          vcmpfalseps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $12, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x0c]
          vcmpneq_oqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $13, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x0d]
          vcmpgeps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $14, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x0e]
          vcmpgtps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $15, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x0f]
          vcmptrueps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $16, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x10]
          vcmpeq_osps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $17, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x11]
          vcmplt_oqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $18, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x12]
          vcmple_oqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $19, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x13]
          vcmpunord_sps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $20, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x14]
          vcmpneq_usps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $21, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x15]
          vcmpnlt_uqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $22, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x16]
          vcmpnle_uqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $23, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x17]
          vcmpord_sps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $24, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x18]
          vcmpeq_usps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $25, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x19]
          vcmpnge_uqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $26, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x1a]
          vcmpngt_uqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $27, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x1b]
          vcmpfalse_osps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $28, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x1c]
          vcmpneq_osps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $29, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x1d]
          vcmpge_oqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $30, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x1e]
          vcmpgt_oqps %ymm1, %ymm2, %ymm3

// CHECK: vcmpps  $31, %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xec,0xc2,0xd9,0x1f]
          vcmptrue_usps %ymm1, %ymm2, %ymm3

// CHECK: vaddsubps  %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xef,0xd0,0xd9]
          vaddsubps  %ymm1, %ymm2, %ymm3

// CHECK: vaddsubps  (%eax), %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xf7,0xd0,0x10]
          vaddsubps  (%eax), %ymm1, %ymm2

// CHECK: vaddsubpd  %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0xd0,0xd9]
          vaddsubpd  %ymm1, %ymm2, %ymm3

// CHECK: vaddsubpd  (%eax), %ymm1, %ymm2
// CHECK: encoding: [0xc5,0xf5,0xd0,0x10]
          vaddsubpd  (%eax), %ymm1, %ymm2

// CHECK: vhaddps  %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xef,0x7c,0xd9]
          vhaddps  %ymm1, %ymm2, %ymm3

// CHECK: vhaddps  (%eax), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xef,0x7c,0x18]
          vhaddps  (%eax), %ymm2, %ymm3

// CHECK: vhaddpd  %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0x7c,0xd9]
          vhaddpd  %ymm1, %ymm2, %ymm3

// CHECK: vhaddpd  (%eax), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0x7c,0x18]
          vhaddpd  (%eax), %ymm2, %ymm3

// CHECK: vhsubps  %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xef,0x7d,0xd9]
          vhsubps  %ymm1, %ymm2, %ymm3

// CHECK: vhsubps  (%eax), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xef,0x7d,0x18]
          vhsubps  (%eax), %ymm2, %ymm3

// CHECK: vhsubpd  %ymm1, %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0x7d,0xd9]
          vhsubpd  %ymm1, %ymm2, %ymm3

// CHECK: vhsubpd  (%eax), %ymm2, %ymm3
// CHECK: encoding: [0xc5,0xed,0x7d,0x18]
          vhsubpd  (%eax), %ymm2, %ymm3

// CHECK: vblendps  $3, %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x55,0x0c,0xca,0x03]
          vblendps  $3, %ymm2, %ymm5, %ymm1

// CHECK: vblendps  $3, (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x55,0x0c,0x08,0x03]
          vblendps  $3, (%eax), %ymm5, %ymm1

// CHECK: vblendpd  $3, %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x55,0x0d,0xca,0x03]
          vblendpd  $3, %ymm2, %ymm5, %ymm1

// CHECK: vblendpd  $3, (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x55,0x0d,0x08,0x03]
          vblendpd  $3, (%eax), %ymm5, %ymm1

// CHECK: vdpps  $3, %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x55,0x40,0xca,0x03]
          vdpps  $3, %ymm2, %ymm5, %ymm1

// CHECK: vdpps  $3, (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x55,0x40,0x08,0x03]
          vdpps  $3, (%eax), %ymm5, %ymm1

// CHECK: vbroadcastf128  (%eax), %ymm2
// CHECK: encoding: [0xc4,0xe2,0x7d,0x1a,0x10]
          vbroadcastf128  (%eax), %ymm2

// CHECK: vbroadcastsd  (%eax), %ymm2
// CHECK: encoding: [0xc4,0xe2,0x7d,0x19,0x10]
          vbroadcastsd  (%eax), %ymm2

// CHECK: vbroadcastss  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x18,0x10]
          vbroadcastss  (%eax), %xmm2

// CHECK: vbroadcastss  (%eax), %ymm2
// CHECK: encoding: [0xc4,0xe2,0x7d,0x18,0x10]
          vbroadcastss  (%eax), %ymm2

// CHECK: vinsertf128  $7, %xmm2, %ymm2, %ymm5
// CHECK: encoding: [0xc4,0xe3,0x6d,0x18,0xea,0x07]
          vinsertf128  $7, %xmm2, %ymm2, %ymm5

// CHECK: vinsertf128  $7, (%eax), %ymm2, %ymm5
// CHECK: encoding: [0xc4,0xe3,0x6d,0x18,0x28,0x07]
          vinsertf128  $7, (%eax), %ymm2, %ymm5

// CHECK: vextractf128  $7, %ymm2, %xmm2
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0xd2,0x07]
          vextractf128  $7, %ymm2, %xmm2

// CHECK: vextractf128  $7, %ymm2, (%eax)
// CHECK: encoding: [0xc4,0xe3,0x7d,0x19,0x10,0x07]
          vextractf128  $7, %ymm2, (%eax)

// CHECK: vmaskmovpd  %xmm2, %xmm5, (%eax)
// CHECK: encoding: [0xc4,0xe2,0x51,0x2f,0x10]
          vmaskmovpd  %xmm2, %xmm5, (%eax)

// CHECK: vmaskmovpd  %ymm2, %ymm5, (%eax)
// CHECK: encoding: [0xc4,0xe2,0x55,0x2f,0x10]
          vmaskmovpd  %ymm2, %ymm5, (%eax)

// CHECK: vmaskmovpd  (%eax), %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x69,0x2d,0x28]
          vmaskmovpd  (%eax), %xmm2, %xmm5

// CHECK: vmaskmovpd  (%eax), %ymm2, %ymm5
// CHECK: encoding: [0xc4,0xe2,0x6d,0x2d,0x28]
          vmaskmovpd  (%eax), %ymm2, %ymm5

// CHECK: vmaskmovps  %xmm2, %xmm5, (%eax)
// CHECK: encoding: [0xc4,0xe2,0x51,0x2e,0x10]
          vmaskmovps  %xmm2, %xmm5, (%eax)

// CHECK: vmaskmovps  %ymm2, %ymm5, (%eax)
// CHECK: encoding: [0xc4,0xe2,0x55,0x2e,0x10]
          vmaskmovps  %ymm2, %ymm5, (%eax)

// CHECK: vmaskmovps  (%eax), %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x69,0x2c,0x28]
          vmaskmovps  (%eax), %xmm2, %xmm5

// CHECK: vmaskmovps  (%eax), %ymm2, %ymm5
// CHECK: encoding: [0xc4,0xe2,0x6d,0x2c,0x28]
          vmaskmovps  (%eax), %ymm2, %ymm5

// CHECK: vpermilps  $7, %xmm1, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0xe9,0x07]
          vpermilps  $7, %xmm1, %xmm5

// CHECK: vpermilps  $7, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0xcd,0x07]
          vpermilps  $7, %ymm5, %ymm1

// CHECK: vpermilps  $7, (%eax), %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x04,0x28,0x07]
          vpermilps  $7, (%eax), %xmm5

// CHECK: vpermilps  $7, (%eax), %ymm5
// CHECK: encoding: [0xc4,0xe3,0x7d,0x04,0x28,0x07]
          vpermilps  $7, (%eax), %ymm5

// CHECK: vpermilps  %xmm1, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x0c,0xc9]
          vpermilps  %xmm1, %xmm5, %xmm1

// CHECK: vpermilps  %ymm1, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x0c,0xc9]
          vpermilps  %ymm1, %ymm5, %ymm1

// CHECK: vpermilps  (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x51,0x0c,0x18]
          vpermilps  (%eax), %xmm5, %xmm3

// CHECK: vpermilps  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x0c,0x08]
          vpermilps  (%eax), %ymm5, %ymm1

// CHECK: vpermilpd  $7, %xmm1, %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0xe9,0x07]
          vpermilpd  $7, %xmm1, %xmm5

// CHECK: vpermilpd  $7, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0xcd,0x07]
          vpermilpd  $7, %ymm5, %ymm1

// CHECK: vpermilpd  $7, (%eax), %xmm5
// CHECK: encoding: [0xc4,0xe3,0x79,0x05,0x28,0x07]
          vpermilpd  $7, (%eax), %xmm5

// CHECK: vpermilpd  $7, (%eax), %ymm5
// CHECK: encoding: [0xc4,0xe3,0x7d,0x05,0x28,0x07]
          vpermilpd  $7, (%eax), %ymm5

// CHECK: vpermilpd  %xmm1, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x51,0x0d,0xc9]
          vpermilpd  %xmm1, %xmm5, %xmm1

// CHECK: vpermilpd  %ymm1, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x0d,0xc9]
          vpermilpd  %ymm1, %ymm5, %ymm1

// CHECK: vpermilpd  (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe2,0x51,0x0d,0x18]
          vpermilpd  (%eax), %xmm5, %xmm3

// CHECK: vpermilpd  (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x55,0x0d,0x08]
          vpermilpd  (%eax), %ymm5, %ymm1

// CHECK: vperm2f128  $7, %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x55,0x06,0xca,0x07]
          vperm2f128  $7, %ymm2, %ymm5, %ymm1

// CHECK: vperm2f128  $7, (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x55,0x06,0x08,0x07]
          vperm2f128  $7, (%eax), %ymm5, %ymm1

// CHECK: vzeroall
// CHECK: encoding: [0xc5,0xfc,0x77]
          vzeroall

// CHECK: vzeroupper
// CHECK: encoding: [0xc5,0xf8,0x77]
          vzeroupper

// CHECK: vcvtsd2sil  %xmm4, %ecx
// CHECK: encoding: [0xc5,0xfb,0x2d,0xcc]
          vcvtsd2sil  %xmm4, %ecx

// CHECK: vcvtsd2sil  (%ecx), %ecx
// CHECK: encoding: [0xc5,0xfb,0x2d,0x09]
          vcvtsd2sil  (%ecx), %ecx

// CHECK: vcvtsi2sd  (%ebp), %xmm0, %xmm7
// CHECK: encoding: [0xc5,0xfb,0x2a,0x7d,0x00]
          vcvtsi2sd  (%ebp), %xmm0, %xmm7

// CHECK: vcvtsi2sd  (%esp), %xmm0, %xmm7
// CHECK: encoding: [0xc5,0xfb,0x2a,0x3c,0x24]
          vcvtsi2sd  (%esp), %xmm0, %xmm7

// CHECK: vlddqu  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xff,0xf0,0x10]
          vlddqu  (%eax), %ymm2

// CHECK: vmovddup  %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xff,0x12,0xea]
          vmovddup  %ymm2, %ymm5

// CHECK: vmovddup  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xff,0x12,0x10]
          vmovddup  (%eax), %ymm2

// CHECK: vmovdqa  %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xfd,0x6f,0xea]
          vmovdqa  %ymm2, %ymm5

// CHECK: vmovdqa  %ymm2, (%eax)
// CHECK: encoding: [0xc5,0xfd,0x7f,0x10]
          vmovdqa  %ymm2, (%eax)

// CHECK: vmovdqa  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfd,0x6f,0x10]
          vmovdqa  (%eax), %ymm2

// CHECK: vmovdqu  %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xfe,0x6f,0xea]
          vmovdqu  %ymm2, %ymm5

// CHECK: vmovdqu  %ymm2, (%eax)
// CHECK: encoding: [0xc5,0xfe,0x7f,0x10]
          vmovdqu  %ymm2, (%eax)

// CHECK: vmovdqu  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfe,0x6f,0x10]
          vmovdqu  (%eax), %ymm2

// CHECK: vmovshdup  %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xfe,0x16,0xea]
          vmovshdup  %ymm2, %ymm5

// CHECK: vmovshdup  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfe,0x16,0x10]
          vmovshdup  (%eax), %ymm2

// CHECK: vmovsldup  %ymm2, %ymm5
// CHECK: encoding: [0xc5,0xfe,0x12,0xea]
          vmovsldup  %ymm2, %ymm5

// CHECK: vmovsldup  (%eax), %ymm2
// CHECK: encoding: [0xc5,0xfe,0x12,0x10]
          vmovsldup  (%eax), %ymm2

// CHECK: vptest  %ymm2, %ymm5
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0xea]
          vptest  %ymm2, %ymm5

// CHECK: vptest  (%eax), %ymm2
// CHECK: encoding: [0xc4,0xe2,0x7d,0x17,0x10]
          vptest  (%eax), %ymm2

// CHECK: vroundpd  $7, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0xcd,0x07]
          vroundpd  $7, %ymm5, %ymm1

// CHECK: vroundpd  $7, (%eax), %ymm5
// CHECK: encoding: [0xc4,0xe3,0x7d,0x09,0x28,0x07]
          vroundpd  $7, (%eax), %ymm5

// CHECK: vroundps  $7, %ymm5, %ymm1
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0xcd,0x07]
          vroundps  $7, %ymm5, %ymm1

// CHECK: vroundps  $7, (%eax), %ymm5
// CHECK: encoding: [0xc4,0xe3,0x7d,0x08,0x28,0x07]
          vroundps  $7, (%eax), %ymm5

// CHECK: vshufpd  $7, %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc5,0xd5,0xc6,0xca,0x07]
          vshufpd  $7, %ymm2, %ymm5, %ymm1

// CHECK: vshufpd  $7, (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc5,0xd5,0xc6,0x08,0x07]
          vshufpd  $7, (%eax), %ymm5, %ymm1

// CHECK: vshufps  $7, %ymm2, %ymm5, %ymm1
// CHECK: encoding: [0xc5,0xd4,0xc6,0xca,0x07]
          vshufps  $7, %ymm2, %ymm5, %ymm1

// CHECK: vshufps  $7, (%eax), %ymm5, %ymm1
// CHECK: encoding: [0xc5,0xd4,0xc6,0x08,0x07]
          vshufps  $7, (%eax), %ymm5, %ymm1

// CHECK: vtestpd  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0xea]
          vtestpd  %xmm2, %xmm5

// CHECK: vtestpd  %ymm2, %ymm5
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0xea]
          vtestpd  %ymm2, %ymm5

// CHECK: vtestpd  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x0f,0x10]
          vtestpd  (%eax), %xmm2

// CHECK: vtestpd  (%eax), %ymm2
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0f,0x10]
          vtestpd  (%eax), %ymm2

// CHECK: vtestps  %xmm2, %xmm5
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0xea]
          vtestps  %xmm2, %xmm5

// CHECK: vtestps  %ymm2, %ymm5
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0xea]
          vtestps  %ymm2, %ymm5

// CHECK: vtestps  (%eax), %xmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x0e,0x10]
          vtestps  (%eax), %xmm2

// CHECK: vtestps  (%eax), %ymm2
// CHECK: encoding: [0xc4,0xe2,0x7d,0x0e,0x10]
          vtestps  (%eax), %ymm2

// CHECK: vblendvpd  %ymm0, 57005(%eax,%eiz), %ymm1, %ymm2
// CHECK: encoding: [0xc4,0xe3,0x75,0x4b,0x94,0x20,0xad,0xde,0x00,0x00,0x00]
          vblendvpd  %ymm0, 0xdead(%eax,%eiz), %ymm1, %ymm2



// CHECK: vpclmulqdq  $17, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x11]
          vpclmulhqhqdq %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $17, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x11]
          vpclmulhqhqdq (%eax), %xmm5, %xmm3

// CHECK: vpclmulqdq  $1, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x01]
          vpclmulhqlqdq %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $1, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x01]
          vpclmulhqlqdq (%eax), %xmm5, %xmm3

// CHECK: vpclmulqdq  $16, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x10]
          vpclmullqhqdq %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $16, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x10]
          vpclmullqhqdq (%eax), %xmm5, %xmm3

// CHECK: vpclmulqdq  $0, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x00]
          vpclmullqlqdq %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $0, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x00]
          vpclmullqlqdq (%eax), %xmm5, %xmm3

// CHECK: vpclmulqdq  $17, %xmm2, %xmm5, %xmm1
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0xca,0x11]
          vpclmulqdq  $17, %xmm2, %xmm5, %xmm1

// CHECK: vpclmulqdq  $17, (%eax), %xmm5, %xmm3
// CHECK: encoding: [0xc4,0xe3,0x51,0x44,0x18,0x11]
          vpclmulqdq  $17, (%eax), %xmm5, %xmm3

// rdar://9795008
// These instructions take a mask not an 8-bit sign extended value.
// CHECK: vblendps $129, %ymm2, %ymm5, %ymm1
          vblendps $0x81, %ymm2, %ymm5, %ymm1
// CHECK: vblendps $129, (%eax), %ymm5, %ymm1
          vblendps $0x81, (%eax), %ymm5, %ymm1
// CHECK: vblendpd $129, %ymm2, %ymm5, %ymm1
          vblendpd $0x81, %ymm2, %ymm5, %ymm1
// CHECK: vblendpd $129, (%eax), %ymm5, %ymm1
          vblendpd $0x81, (%eax), %ymm5, %ymm1
// CHECK: vpblendw $129, %xmm2, %xmm5, %xmm1
          vpblendw $0x81, %xmm2, %xmm5, %xmm1
// CHECK: vmpsadbw $129, %xmm2, %xmm5, %xmm1
          vmpsadbw $0x81, %xmm2, %xmm5, %xmm1
// CHECK: vdpps $129, %ymm2, %ymm5, %ymm1
          vdpps $0x81, %ymm2, %ymm5, %ymm1
// CHECK: vdpps $129, (%eax), %ymm5, %ymm1
          vdpps $0x81, (%eax), %ymm5, %ymm1
// CHECK: vdppd $129, %xmm2, %xmm5, %xmm1
          vdppd $0x81, %xmm2, %xmm5, %xmm1
// CHECK: vinsertps $129, %xmm3, %xmm2, %xmm1
          vinsertps $0x81, %xmm3, %xmm2, %xmm1
