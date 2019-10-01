// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vaddss  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x32,0x58,0xd0]
vaddss  %xmm8, %xmm9, %xmm10

// CHECK: vmulss  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x32,0x59,0xd0]
vmulss  %xmm8, %xmm9, %xmm10

// CHECK: vsubss  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x32,0x5c,0xd0]
vsubss  %xmm8, %xmm9, %xmm10

// CHECK: vdivss  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x32,0x5e,0xd0]
vdivss  %xmm8, %xmm9, %xmm10

// CHECK: vaddsd  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x33,0x58,0xd0]
vaddsd  %xmm8, %xmm9, %xmm10

// CHECK: vmulsd  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x33,0x59,0xd0]
vmulsd  %xmm8, %xmm9, %xmm10

// CHECK: vsubsd  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x33,0x5c,0xd0]
vsubsd  %xmm8, %xmm9, %xmm10

// CHECK: vdivsd  %xmm8, %xmm9, %xmm10
// CHECK:  encoding: [0xc4,0x41,0x33,0x5e,0xd0]
vdivsd  %xmm8, %xmm9, %xmm10

// CHECK:   vaddss  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2a,0x58,0x5c,0xd9,0xfc]
vaddss  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vsubss  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2a,0x5c,0x5c,0xd9,0xfc]
vsubss  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vmulss  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2a,0x59,0x5c,0xd9,0xfc]
vmulss  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vdivss  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2a,0x5e,0x5c,0xd9,0xfc]
vdivss  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vaddsd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2b,0x58,0x5c,0xd9,0xfc]
vaddsd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vsubsd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2b,0x5c,0x5c,0xd9,0xfc]
vsubsd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vmulsd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2b,0x59,0x5c,0xd9,0xfc]
vmulsd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK:   vdivsd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK:   encoding: [0xc5,0x2b,0x5e,0x5c,0xd9,0xfc]
vdivsd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vaddps  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x20,0x58,0xfa]
vaddps  %xmm10, %xmm11, %xmm15

// CHECK: vsubps  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x20,0x5c,0xfa]
vsubps  %xmm10, %xmm11, %xmm15

// CHECK: vmulps  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x20,0x59,0xfa]
vmulps  %xmm10, %xmm11, %xmm15

// CHECK: vdivps  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x20,0x5e,0xfa]
vdivps  %xmm10, %xmm11, %xmm15

// CHECK: vaddpd  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x21,0x58,0xfa]
vaddpd  %xmm10, %xmm11, %xmm15

// CHECK: vsubpd  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x21,0x5c,0xfa]
vsubpd  %xmm10, %xmm11, %xmm15

// CHECK: vmulpd  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x21,0x59,0xfa]
vmulpd  %xmm10, %xmm11, %xmm15

// CHECK: vdivpd  %xmm10, %xmm11, %xmm15
// CHECK: encoding: [0xc4,0x41,0x21,0x5e,0xfa]
vdivpd  %xmm10, %xmm11, %xmm15

// CHECK: vaddps  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x28,0x58,0x5c,0xd9,0xfc]
vaddps  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vsubps  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x28,0x5c,0x5c,0xd9,0xfc]
vsubps  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vmulps  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x28,0x59,0x5c,0xd9,0xfc]
vmulps  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vdivps  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x28,0x5e,0x5c,0xd9,0xfc]
vdivps  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vaddpd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x29,0x58,0x5c,0xd9,0xfc]
vaddpd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vsubpd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x29,0x5c,0x5c,0xd9,0xfc]
vsubpd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vmulpd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x29,0x59,0x5c,0xd9,0xfc]
vmulpd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vdivpd  -4(%rcx,%rbx,8), %xmm10, %xmm11
// CHECK: encoding: [0xc5,0x29,0x5e,0x5c,0xd9,0xfc]
vdivpd  -4(%rcx,%rbx,8), %xmm10, %xmm11

// CHECK: vmaxss  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x0a,0x5f,0xe2]
          vmaxss  %xmm10, %xmm14, %xmm12

// CHECK: vmaxsd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x0b,0x5f,0xe2]
          vmaxsd  %xmm10, %xmm14, %xmm12

// CHECK: vminss  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x0a,0x5d,0xe2]
          vminss  %xmm10, %xmm14, %xmm12

// CHECK: vminsd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x0b,0x5d,0xe2]
          vminsd  %xmm10, %xmm14, %xmm12

// CHECK: vmaxss  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1a,0x5f,0x54,0xcb,0xfc]
          vmaxss  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vmaxsd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1b,0x5f,0x54,0xcb,0xfc]
          vmaxsd  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vminss  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1a,0x5d,0x54,0xcb,0xfc]
          vminss  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vminsd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1b,0x5d,0x54,0xcb,0xfc]
          vminsd  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vmaxps  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x08,0x5f,0xe2]
          vmaxps  %xmm10, %xmm14, %xmm12

// CHECK: vmaxpd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x09,0x5f,0xe2]
          vmaxpd  %xmm10, %xmm14, %xmm12

// CHECK: vminps  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x08,0x5d,0xe2]
          vminps  %xmm10, %xmm14, %xmm12

// CHECK: vminpd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x09,0x5d,0xe2]
          vminpd  %xmm10, %xmm14, %xmm12

// CHECK: vmaxps  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x18,0x5f,0x54,0xcb,0xfc]
          vmaxps  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vmaxpd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x19,0x5f,0x54,0xcb,0xfc]
          vmaxpd  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vminps  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x18,0x5d,0x54,0xcb,0xfc]
          vminps  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vminpd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x19,0x5d,0x54,0xcb,0xfc]
          vminpd  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vandps  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x08,0x54,0xe2]
          vandps  %xmm10, %xmm14, %xmm12

// CHECK: vandpd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x09,0x54,0xe2]
          vandpd  %xmm10, %xmm14, %xmm12

// CHECK: vandps  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x18,0x54,0x54,0xcb,0xfc]
          vandps  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vandpd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x19,0x54,0x54,0xcb,0xfc]
          vandpd  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vorps  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x08,0x56,0xe2]
          vorps  %xmm10, %xmm14, %xmm12

// CHECK: vorpd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x09,0x56,0xe2]
          vorpd  %xmm10, %xmm14, %xmm12

// CHECK: vorps  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x18,0x56,0x54,0xcb,0xfc]
          vorps  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vorpd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x19,0x56,0x54,0xcb,0xfc]
          vorpd  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vxorps  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x08,0x57,0xe2]
          vxorps  %xmm10, %xmm14, %xmm12

// CHECK: vxorpd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x09,0x57,0xe2]
          vxorpd  %xmm10, %xmm14, %xmm12

// CHECK: vxorps  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x18,0x57,0x54,0xcb,0xfc]
          vxorps  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vxorpd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x19,0x57,0x54,0xcb,0xfc]
          vxorpd  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vandnps  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x08,0x55,0xe2]
          vandnps  %xmm10, %xmm14, %xmm12

// CHECK: vandnpd  %xmm10, %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x09,0x55,0xe2]
          vandnpd  %xmm10, %xmm14, %xmm12

// CHECK: vandnps  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x18,0x55,0x54,0xcb,0xfc]
          vandnps  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vandnpd  -4(%rbx,%rcx,8), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x19,0x55,0x54,0xcb,0xfc]
          vandnpd  -4(%rbx,%rcx,8), %xmm12, %xmm10

// CHECK: vmovss  -4(%rbx,%rcx,8), %xmm10
// CHECK: encoding: [0xc5,0x7a,0x10,0x54,0xcb,0xfc]
          vmovss  -4(%rbx,%rcx,8), %xmm10

// CHECK: vmovss  %xmm14, %xmm10, %xmm15
// CHECK: encoding: [0xc4,0x41,0x2a,0x10,0xfe]
          vmovss  %xmm14, %xmm10, %xmm15

// CHECK: vmovsd  -4(%rbx,%rcx,8), %xmm10
// CHECK: encoding: [0xc5,0x7b,0x10,0x54,0xcb,0xfc]
          vmovsd  -4(%rbx,%rcx,8), %xmm10

// CHECK: vmovsd  %xmm14, %xmm10, %xmm15
// CHECK: encoding: [0xc4,0x41,0x2b,0x10,0xfe]
          vmovsd  %xmm14, %xmm10, %xmm15

// CHECK: vunpckhps  %xmm15, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0x15,0xef]
          vunpckhps  %xmm15, %xmm12, %xmm13

// CHECK: vunpckhpd  %xmm15, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x15,0xef]
          vunpckhpd  %xmm15, %xmm12, %xmm13

// CHECK: vunpcklps  %xmm15, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0x14,0xef]
          vunpcklps  %xmm15, %xmm12, %xmm13

// CHECK: vunpcklpd  %xmm15, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x14,0xef]
          vunpcklpd  %xmm15, %xmm12, %xmm13

// CHECK: vunpckhps  -4(%rbx,%rcx,8), %xmm12, %xmm15
// CHECK: encoding: [0xc5,0x18,0x15,0x7c,0xcb,0xfc]
          vunpckhps  -4(%rbx,%rcx,8), %xmm12, %xmm15

// CHECK: vunpckhpd  -4(%rbx,%rcx,8), %xmm12, %xmm15
// CHECK: encoding: [0xc5,0x19,0x15,0x7c,0xcb,0xfc]
          vunpckhpd  -4(%rbx,%rcx,8), %xmm12, %xmm15

// CHECK: vunpcklps  -4(%rbx,%rcx,8), %xmm12, %xmm15
// CHECK: encoding: [0xc5,0x18,0x14,0x7c,0xcb,0xfc]
          vunpcklps  -4(%rbx,%rcx,8), %xmm12, %xmm15

// CHECK: vunpcklpd  -4(%rbx,%rcx,8), %xmm12, %xmm15
// CHECK: encoding: [0xc5,0x19,0x14,0x7c,0xcb,0xfc]
          vunpcklpd  -4(%rbx,%rcx,8), %xmm12, %xmm15

// CHECK: vcmpeqps  %xmm10, %xmm12, %xmm15
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xfa,0x00]
          vcmpps  $0, %xmm10, %xmm12, %xmm15

// CHECK: vcmpeqps  (%rax), %xmm12, %xmm15
// CHECK: encoding: [0xc5,0x18,0xc2,0x38,0x00]
          vcmpps  $0, (%rax), %xmm12, %xmm15

// CHECK: vcmpordps  %xmm10, %xmm12, %xmm15
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xfa,0x07]
          vcmpps  $7, %xmm10, %xmm12, %xmm15

// CHECK: vcmpeqpd  %xmm10, %xmm12, %xmm15
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xfa,0x00]
          vcmppd  $0, %xmm10, %xmm12, %xmm15

// CHECK: vcmpeqpd  (%rax), %xmm12, %xmm15
// CHECK: encoding: [0xc5,0x19,0xc2,0x38,0x00]
          vcmppd  $0, (%rax), %xmm12, %xmm15

// CHECK: vcmpordpd  %xmm10, %xmm12, %xmm15
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xfa,0x07]
          vcmppd  $7, %xmm10, %xmm12, %xmm15

// CHECK: vshufps  $8, %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc6,0xeb,0x08]
          vshufps  $8, %xmm11, %xmm12, %xmm13

// CHECK: vshufps  $8, -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc6,0x6c,0xcb,0xfc,0x08]
          vshufps  $8, -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vshufpd  $8, %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc6,0xeb,0x08]
          vshufpd  $8, %xmm11, %xmm12, %xmm13

// CHECK: vshufpd  $8, -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc6,0x6c,0xcb,0xfc,0x08]
          vshufpd  $8, -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x00]
          vcmpeqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpleps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x02]
          vcmpleps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpltps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x01]
          vcmpltps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x04]
          vcmpneqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnleps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x06]
          vcmpnleps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnltps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x05]
          vcmpnltps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpordps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x07]
          vcmpordps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpunordps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x03]
          vcmpunordps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x00]
          vcmpeqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpleps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x02]
          vcmpleps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpltps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x01]
          vcmpltps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x04]
          vcmpneqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnleps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x06]
          vcmpnleps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnltps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x05]
          vcmpnltps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpordps  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc8,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordps   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmpunordps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x03]
          vcmpunordps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x00]
          vcmpeqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmplepd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x02]
          vcmplepd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpltpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x01]
          vcmpltpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x04]
          vcmpneqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnlepd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x06]
          vcmpnlepd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnltpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x05]
          vcmpnltpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpordpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x07]
          vcmpordpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpunordpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x03]
          vcmpunordpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x00]
          vcmpeqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmplepd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x02]
          vcmplepd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpltpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x01]
          vcmpltpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x04]
          vcmpneqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnlepd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x06]
          vcmpnlepd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnltpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x05]
          vcmpnltpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpordpd  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc9,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordpd   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmpunordpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x03]
          vcmpunordpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x00]
          vcmpeqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpless  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x02]
          vcmpless   %xmm11, %xmm12, %xmm13

// CHECK: vcmpltss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x01]
          vcmpltss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x04]
          vcmpneqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnless  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x06]
          vcmpnless   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnltss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x05]
          vcmpnltss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpordss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x07]
          vcmpordss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpunordss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x03]
          vcmpunordss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x00]
          vcmpeqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpless  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x02]
          vcmpless   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpltss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x01]
          vcmpltss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x04]
          vcmpneqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnless  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x06]
          vcmpnless   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnltss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x05]
          vcmpnltss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpordss  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xca,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordss   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmpunordss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x03]
          vcmpunordss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x00]
          vcmpeqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmplesd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x02]
          vcmplesd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpltsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x01]
          vcmpltsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x04]
          vcmpneqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnlesd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x06]
          vcmpnlesd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnltsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x05]
          vcmpnltsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpordsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x07]
          vcmpordsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpunordsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x03]
          vcmpunordsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x00]
          vcmpeqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmplesd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x02]
          vcmplesd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpltsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x01]
          vcmpltsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x04]
          vcmpneqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnlesd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x06]
          vcmpnlesd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnltsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x05]
          vcmpnltsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpordsd  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xcb,0xc2,0x54,0xcb,0xfc,0x07]
          vcmpordsd   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmpunordsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x03]
          vcmpunordsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x08]
          vcmpeq_uqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngeps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x09]
          vcmpngeps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngtps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0a]
          vcmpngtps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalseps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0b]
          vcmpfalseps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0c]
          vcmpneq_oqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgeps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0d]
          vcmpgeps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgtps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0e]
          vcmpgtps   %xmm11, %xmm12, %xmm13

// CHECK: vcmptrueps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0f]
          vcmptrueps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_osps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x10]
          vcmpeq_osps   %xmm11, %xmm12, %xmm13

// CHECK: vcmplt_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x11]
          vcmplt_oqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmple_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x12]
          vcmple_oqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpunord_sps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x13]
          vcmpunord_sps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_usps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x14]
          vcmpneq_usps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnlt_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x15]
          vcmpnlt_uqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnle_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x16]
          vcmpnle_uqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpord_sps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x17]
          vcmpord_sps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_usps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x18]
          vcmpeq_usps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnge_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x19]
          vcmpnge_uqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngt_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1a]
          vcmpngt_uqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalse_osps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1b]
          vcmpfalse_osps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_osps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1c]
          vcmpneq_osps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpge_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1d]
          vcmpge_oqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgt_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1e]
          vcmpgt_oqps   %xmm11, %xmm12, %xmm13

// CHECK: vcmptrue_usps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1f]
          vcmptrue_usps   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_uqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x08]
          vcmpeq_uqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngeps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x09]
          vcmpngeps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngtps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x0a]
          vcmpngtps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpfalseps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x0b]
          vcmpfalseps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_oqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x0c]
          vcmpneq_oqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgeps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x0d]
          vcmpgeps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgtps  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc8,0xc2,0x54,0xcb,0xfc,0x0e]
          vcmpgtps   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmptrueps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x0f]
          vcmptrueps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_osps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x10]
          vcmpeq_osps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmplt_oqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x11]
          vcmplt_oqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmple_oqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x12]
          vcmple_oqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpunord_sps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x13]
          vcmpunord_sps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_usps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x14]
          vcmpneq_usps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnlt_uqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x15]
          vcmpnlt_uqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnle_uqps  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc8,0xc2,0x54,0xcb,0xfc,0x16]
          vcmpnle_uqps   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmpord_sps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x17]
          vcmpord_sps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_usps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x18]
          vcmpeq_usps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnge_uqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x19]
          vcmpnge_uqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngt_uqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x1a]
          vcmpngt_uqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpfalse_osps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x1b]
          vcmpfalse_osps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_osps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x1c]
          vcmpneq_osps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpge_oqps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x1d]
          vcmpge_oqps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgt_oqps  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc8,0xc2,0x54,0xcb,0xfc,0x1e]
          vcmpgt_oqps   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmptrue_usps  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0xc2,0x6c,0xcb,0xfc,0x1f]
          vcmptrue_usps   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_uqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x08]
          vcmpeq_uqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngepd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x09]
          vcmpngepd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngtpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x0a]
          vcmpngtpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalsepd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x0b]
          vcmpfalsepd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_oqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x0c]
          vcmpneq_oqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgepd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x0d]
          vcmpgepd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgtpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x0e]
          vcmpgtpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmptruepd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x0f]
          vcmptruepd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_ospd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x10]
          vcmpeq_ospd   %xmm11, %xmm12, %xmm13

// CHECK: vcmplt_oqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x11]
          vcmplt_oqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmple_oqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x12]
          vcmple_oqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpunord_spd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x13]
          vcmpunord_spd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_uspd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x14]
          vcmpneq_uspd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnlt_uqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x15]
          vcmpnlt_uqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnle_uqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x16]
          vcmpnle_uqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpord_spd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x17]
          vcmpord_spd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_uspd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x18]
          vcmpeq_uspd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnge_uqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x19]
          vcmpnge_uqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngt_uqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x1a]
          vcmpngt_uqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalse_ospd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x1b]
          vcmpfalse_ospd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_ospd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x1c]
          vcmpneq_ospd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpge_oqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x1d]
          vcmpge_oqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgt_oqpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x1e]
          vcmpgt_oqpd   %xmm11, %xmm12, %xmm13

// CHECK: vcmptrue_uspd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xc2,0xeb,0x1f]
          vcmptrue_uspd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_uqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x08]
          vcmpeq_uqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngepd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x09]
          vcmpngepd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngtpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x0a]
          vcmpngtpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpfalsepd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x0b]
          vcmpfalsepd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_oqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x0c]
          vcmpneq_oqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgepd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x0d]
          vcmpgepd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgtpd  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc9,0xc2,0x54,0xcb,0xfc,0x0e]
          vcmpgtpd   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmptruepd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x0f]
          vcmptruepd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_ospd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x10]
          vcmpeq_ospd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmplt_oqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x11]
          vcmplt_oqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmple_oqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x12]
          vcmple_oqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpunord_spd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x13]
          vcmpunord_spd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_uspd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x14]
          vcmpneq_uspd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnlt_uqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x15]
          vcmpnlt_uqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnle_uqpd  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc9,0xc2,0x54,0xcb,0xfc,0x16]
          vcmpnle_uqpd   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmpord_spd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x17]
          vcmpord_spd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_uspd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x18]
          vcmpeq_uspd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnge_uqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x19]
          vcmpnge_uqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngt_uqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x1a]
          vcmpngt_uqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpfalse_ospd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x1b]
          vcmpfalse_ospd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_ospd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x1c]
          vcmpneq_ospd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpge_oqpd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x1d]
          vcmpge_oqpd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgt_oqpd  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xc9,0xc2,0x54,0xcb,0xfc,0x1e]
          vcmpgt_oqpd   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmptrue_uspd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc2,0x6c,0xcb,0xfc,0x1f]
          vcmptrue_uspd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_uqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x08]
          vcmpeq_uqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngess  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x09]
          vcmpngess   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngtss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x0a]
          vcmpngtss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalsess  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x0b]
          vcmpfalsess   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_oqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x0c]
          vcmpneq_oqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgess  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x0d]
          vcmpgess   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgtss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x0e]
          vcmpgtss   %xmm11, %xmm12, %xmm13

// CHECK: vcmptruess  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x0f]
          vcmptruess   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_osss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x10]
          vcmpeq_osss   %xmm11, %xmm12, %xmm13

// CHECK: vcmplt_oqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x11]
          vcmplt_oqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmple_oqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x12]
          vcmple_oqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpunord_sss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x13]
          vcmpunord_sss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_usss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x14]
          vcmpneq_usss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnlt_uqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x15]
          vcmpnlt_uqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnle_uqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x16]
          vcmpnle_uqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpord_sss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x17]
          vcmpord_sss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_usss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x18]
          vcmpeq_usss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnge_uqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x19]
          vcmpnge_uqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngt_uqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x1a]
          vcmpngt_uqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalse_osss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x1b]
          vcmpfalse_osss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_osss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x1c]
          vcmpneq_osss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpge_oqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x1d]
          vcmpge_oqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgt_oqss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x1e]
          vcmpgt_oqss   %xmm11, %xmm12, %xmm13

// CHECK: vcmptrue_usss  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1a,0xc2,0xeb,0x1f]
          vcmptrue_usss   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_uqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x08]
          vcmpeq_uqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngess  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x09]
          vcmpngess   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngtss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x0a]
          vcmpngtss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpfalsess  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x0b]
          vcmpfalsess   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_oqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x0c]
          vcmpneq_oqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgess  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x0d]
          vcmpgess   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgtss  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xca,0xc2,0x54,0xcb,0xfc,0x0e]
          vcmpgtss   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmptruess  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x0f]
          vcmptruess   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_osss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x10]
          vcmpeq_osss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmplt_oqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x11]
          vcmplt_oqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmple_oqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x12]
          vcmple_oqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpunord_sss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x13]
          vcmpunord_sss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_usss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x14]
          vcmpneq_usss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnlt_uqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x15]
          vcmpnlt_uqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnle_uqss  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xca,0xc2,0x54,0xcb,0xfc,0x16]
          vcmpnle_uqss   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmpord_sss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x17]
          vcmpord_sss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_usss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x18]
          vcmpeq_usss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnge_uqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x19]
          vcmpnge_uqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngt_uqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x1a]
          vcmpngt_uqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpfalse_osss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x1b]
          vcmpfalse_osss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_osss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x1c]
          vcmpneq_osss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpge_oqss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x1d]
          vcmpge_oqss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgt_oqss  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xca,0xc2,0x54,0xcb,0xfc,0x1e]
          vcmpgt_oqss   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmptrue_usss  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1a,0xc2,0x6c,0xcb,0xfc,0x1f]
          vcmptrue_usss   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_uqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x08]
          vcmpeq_uqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngesd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x09]
          vcmpngesd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngtsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x0a]
          vcmpngtsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalsesd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x0b]
          vcmpfalsesd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_oqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x0c]
          vcmpneq_oqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgesd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x0d]
          vcmpgesd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgtsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x0e]
          vcmpgtsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmptruesd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x0f]
          vcmptruesd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_ossd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x10]
          vcmpeq_ossd   %xmm11, %xmm12, %xmm13

// CHECK: vcmplt_oqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x11]
          vcmplt_oqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmple_oqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x12]
          vcmple_oqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpunord_ssd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x13]
          vcmpunord_ssd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_ussd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x14]
          vcmpneq_ussd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnlt_uqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x15]
          vcmpnlt_uqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnle_uqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x16]
          vcmpnle_uqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpord_ssd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x17]
          vcmpord_ssd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_ussd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x18]
          vcmpeq_ussd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpnge_uqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x19]
          vcmpnge_uqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpngt_uqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x1a]
          vcmpngt_uqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalse_ossd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x1b]
          vcmpfalse_ossd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_ossd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x1c]
          vcmpneq_ossd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpge_oqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x1d]
          vcmpge_oqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpgt_oqsd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x1e]
          vcmpgt_oqsd   %xmm11, %xmm12, %xmm13

// CHECK: vcmptrue_ussd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xc2,0xeb,0x1f]
          vcmptrue_ussd   %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_uqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x08]
          vcmpeq_uqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngesd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x09]
          vcmpngesd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngtsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x0a]
          vcmpngtsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpfalsesd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x0b]
          vcmpfalsesd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_oqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x0c]
          vcmpneq_oqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgesd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x0d]
          vcmpgesd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgtsd  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xcb,0xc2,0x54,0xcb,0xfc,0x0e]
          vcmpgtsd   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmptruesd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x0f]
          vcmptruesd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_ossd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x10]
          vcmpeq_ossd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmplt_oqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x11]
          vcmplt_oqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmple_oqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x12]
          vcmple_oqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpunord_ssd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x13]
          vcmpunord_ssd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_ussd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x14]
          vcmpneq_ussd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnlt_uqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x15]
          vcmpnlt_uqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnle_uqsd  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xcb,0xc2,0x54,0xcb,0xfc,0x16]
          vcmpnle_uqsd   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmpord_ssd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x17]
          vcmpord_ssd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpeq_ussd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x18]
          vcmpeq_ussd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpnge_uqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x19]
          vcmpnge_uqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpngt_uqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x1a]
          vcmpngt_uqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpfalse_ossd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x1b]
          vcmpfalse_ossd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpneq_ossd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x1c]
          vcmpneq_ossd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpge_oqsd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x1d]
          vcmpge_oqsd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vcmpgt_oqsd  -4(%rbx,%rcx,8), %xmm6, %xmm2
// CHECK: encoding: [0xc5,0xcb,0xc2,0x54,0xcb,0xfc,0x1e]
          vcmpgt_oqsd   -4(%rbx,%rcx,8), %xmm6, %xmm2

// CHECK: vcmptrue_ussd  -4(%rbx,%rcx,8), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0xc2,0x6c,0xcb,0xfc,0x1f]
          vcmptrue_ussd   -4(%rbx,%rcx,8), %xmm12, %xmm13

// CHECK: vucomiss  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x78,0x2e,0xe3]
          vucomiss  %xmm11, %xmm12

// CHECK: vucomiss  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x78,0x2e,0x20]
          vucomiss  (%rax), %xmm12

// CHECK: vcomiss  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x78,0x2f,0xe3]
          vcomiss  %xmm11, %xmm12

// CHECK: vcomiss  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x78,0x2f,0x20]
          vcomiss  (%rax), %xmm12

// CHECK: vucomisd  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x79,0x2e,0xe3]
          vucomisd  %xmm11, %xmm12

// CHECK: vucomisd  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x79,0x2e,0x20]
          vucomisd  (%rax), %xmm12

// CHECK: vcomisd  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x79,0x2f,0xe3]
          vcomisd  %xmm11, %xmm12

// CHECK: vcomisd  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x79,0x2f,0x20]
          vcomisd  (%rax), %xmm12

// CHECK: vcvttss2si  (%rcx), %eax
// CHECK: encoding: [0xc5,0xfa,0x2c,0x01]
          vcvttss2si  (%rcx), %eax

// CHECK: vcvtsi2ssl  (%rax), %xmm11, %xmm12
// CHECK: encoding: [0xc5,0x22,0x2a,0x20]
          vcvtsi2ssl  (%rax), %xmm11, %xmm12

// CHECK: vcvtsi2ssl  (%rax), %xmm11, %xmm12
// CHECK: encoding: [0xc5,0x22,0x2a,0x20]
          vcvtsi2ssl  (%rax), %xmm11, %xmm12

// CHECK: vcvttsd2si  (%rcx), %eax
// CHECK: encoding: [0xc5,0xfb,0x2c,0x01]
          vcvttsd2si  (%rcx), %eax

// CHECK: vcvtsi2sdl  (%rax), %xmm11, %xmm12
// CHECK: encoding: [0xc5,0x23,0x2a,0x20]
          vcvtsi2sdl  (%rax), %xmm11, %xmm12

// CHECK: vcvtsi2sdl  (%rax), %xmm11, %xmm12
// CHECK: encoding: [0xc5,0x23,0x2a,0x20]
          vcvtsi2sdl  (%rax), %xmm11, %xmm12

// CHECK: vmovaps  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x78,0x28,0x20]
          vmovaps  (%rax), %xmm12

// CHECK: vmovaps  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x78,0x28,0xe3]
          vmovaps  %xmm11, %xmm12

// CHECK: vmovaps  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x78,0x29,0x18]
          vmovaps  %xmm11, (%rax)

// CHECK: vmovapd  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x79,0x28,0x20]
          vmovapd  (%rax), %xmm12

// CHECK: vmovapd  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x79,0x28,0xe3]
          vmovapd  %xmm11, %xmm12

// CHECK: vmovapd  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x79,0x29,0x18]
          vmovapd  %xmm11, (%rax)

// CHECK: vmovups  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x78,0x10,0x20]
          vmovups  (%rax), %xmm12

// CHECK: vmovups  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x78,0x10,0xe3]
          vmovups  %xmm11, %xmm12

// CHECK: vmovups  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x78,0x11,0x18]
          vmovups  %xmm11, (%rax)

// CHECK: vmovupd  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x79,0x10,0x20]
          vmovupd  (%rax), %xmm12

// CHECK: vmovupd  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x79,0x10,0xe3]
          vmovupd  %xmm11, %xmm12

// CHECK: vmovupd  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x79,0x11,0x18]
          vmovupd  %xmm11, (%rax)

// CHECK: vmovlps  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x78,0x13,0x18]
          vmovlps  %xmm11, (%rax)

// CHECK: vmovlps  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0x12,0x28]
          vmovlps  (%rax), %xmm12, %xmm13

// CHECK: vmovlpd  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x79,0x13,0x18]
          vmovlpd  %xmm11, (%rax)

// CHECK: vmovlpd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x12,0x28]
          vmovlpd  (%rax), %xmm12, %xmm13

// CHECK: vmovhps  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x78,0x17,0x18]
          vmovhps  %xmm11, (%rax)

// CHECK: vmovhps  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x18,0x16,0x28]
          vmovhps  (%rax), %xmm12, %xmm13

// CHECK: vmovhpd  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x79,0x17,0x18]
          vmovhpd  %xmm11, (%rax)

// CHECK: vmovhpd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x16,0x28]
          vmovhpd  (%rax), %xmm12, %xmm13

// CHECK: vmovlhps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0x16,0xeb]
          vmovlhps  %xmm11, %xmm12, %xmm13

// CHECK: vmovhlps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0x12,0xeb]
          vmovhlps  %xmm11, %xmm12, %xmm13

// CHECK: vcvtss2si  %xmm11, %eax
// CHECK: encoding: [0xc4,0xc1,0x7a,0x2d,0xc3]
          vcvtss2si  %xmm11, %eax

// CHECK: vcvtss2si  (%rax), %ebx
// CHECK: encoding: [0xc5,0xfa,0x2d,0x18]
          vcvtss2si  (%rax), %ebx

// CHECK: vcvtdq2ps  %xmm10, %xmm12
// CHECK: encoding: [0xc4,0x41,0x78,0x5b,0xe2]
          vcvtdq2ps  %xmm10, %xmm12

// CHECK: vcvtdq2ps  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x78,0x5b,0x20]
          vcvtdq2ps  (%rax), %xmm12

// CHECK: vcvtsd2ss  %xmm12, %xmm13, %xmm10
// CHECK: encoding: [0xc4,0x41,0x13,0x5a,0xd4]
          vcvtsd2ss  %xmm12, %xmm13, %xmm10

// CHECK: vcvtsd2ss  (%rax), %xmm13, %xmm10
// CHECK: encoding: [0xc5,0x13,0x5a,0x10]
          vcvtsd2ss  (%rax), %xmm13, %xmm10

// CHECK: vcvtps2dq  %xmm12, %xmm11
// CHECK: encoding: [0xc4,0x41,0x79,0x5b,0xdc]
          vcvtps2dq  %xmm12, %xmm11

// CHECK: vcvtps2dq  (%rax), %xmm11
// CHECK: encoding: [0xc5,0x79,0x5b,0x18]
          vcvtps2dq  (%rax), %xmm11

// CHECK: vcvtss2sd  %xmm12, %xmm13, %xmm10
// CHECK: encoding: [0xc4,0x41,0x12,0x5a,0xd4]
          vcvtss2sd  %xmm12, %xmm13, %xmm10

// CHECK: vcvtss2sd  (%rax), %xmm13, %xmm10
// CHECK: encoding: [0xc5,0x12,0x5a,0x10]
          vcvtss2sd  (%rax), %xmm13, %xmm10

// CHECK: vcvtdq2ps  %xmm13, %xmm10
// CHECK: encoding: [0xc4,0x41,0x78,0x5b,0xd5]
          vcvtdq2ps  %xmm13, %xmm10

// CHECK: vcvtdq2ps  (%ecx), %xmm13
// CHECK: encoding: [0x67,0xc5,0x78,0x5b,0x29]
          vcvtdq2ps  (%ecx), %xmm13

// CHECK: vcvttps2dq  %xmm12, %xmm11
// CHECK: encoding: [0xc4,0x41,0x7a,0x5b,0xdc]
          vcvttps2dq  %xmm12, %xmm11

// CHECK: vcvttps2dq  (%rax), %xmm11
// CHECK: encoding: [0xc5,0x7a,0x5b,0x18]
          vcvttps2dq  (%rax), %xmm11

// CHECK: vcvtps2pd  %xmm12, %xmm11
// CHECK: encoding: [0xc4,0x41,0x78,0x5a,0xdc]
          vcvtps2pd  %xmm12, %xmm11

// CHECK: vcvtps2pd  (%rax), %xmm11
// CHECK: encoding: [0xc5,0x78,0x5a,0x18]
          vcvtps2pd  (%rax), %xmm11

// CHECK: vcvtpd2ps  %xmm12, %xmm11
// CHECK: encoding: [0xc4,0x41,0x79,0x5a,0xdc]
          vcvtpd2ps  %xmm12, %xmm11

// CHECK: vsqrtpd  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x79,0x51,0xe3]
          vsqrtpd  %xmm11, %xmm12

// CHECK: vsqrtpd  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x79,0x51,0x20]
          vsqrtpd  (%rax), %xmm12

// CHECK: vsqrtps  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x78,0x51,0xe3]
          vsqrtps  %xmm11, %xmm12

// CHECK: vsqrtps  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x78,0x51,0x20]
          vsqrtps  (%rax), %xmm12

// CHECK: vsqrtsd  %xmm11, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x41,0x1b,0x51,0xd3]
          vsqrtsd  %xmm11, %xmm12, %xmm10

// CHECK: vsqrtsd  (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1b,0x51,0x10]
          vsqrtsd  (%rax), %xmm12, %xmm10

// CHECK: vsqrtss  %xmm11, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x41,0x1a,0x51,0xd3]
          vsqrtss  %xmm11, %xmm12, %xmm10

// CHECK: vsqrtss  (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1a,0x51,0x10]
          vsqrtss  (%rax), %xmm12, %xmm10

// CHECK: vrsqrtps  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x78,0x52,0xe3]
          vrsqrtps  %xmm11, %xmm12

// CHECK: vrsqrtps  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x78,0x52,0x20]
          vrsqrtps  (%rax), %xmm12

// CHECK: vrsqrtss  %xmm11, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x41,0x1a,0x52,0xd3]
          vrsqrtss  %xmm11, %xmm12, %xmm10

// CHECK: vrsqrtss  (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1a,0x52,0x10]
          vrsqrtss  (%rax), %xmm12, %xmm10

// CHECK: vrcpps  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x78,0x53,0xe3]
          vrcpps  %xmm11, %xmm12

// CHECK: vrcpps  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x78,0x53,0x20]
          vrcpps  (%rax), %xmm12

// CHECK: vrcpss  %xmm11, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x41,0x1a,0x53,0xd3]
          vrcpss  %xmm11, %xmm12, %xmm10

// CHECK: vrcpss  (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x1a,0x53,0x10]
          vrcpss  (%rax), %xmm12, %xmm10

// CHECK: vmovntdq  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x79,0xe7,0x18]
          vmovntdq  %xmm11, (%rax)

// CHECK: vmovntpd  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x79,0x2b,0x18]
          vmovntpd  %xmm11, (%rax)

// CHECK: vmovntps  %xmm11, (%rax)
// CHECK: encoding: [0xc5,0x78,0x2b,0x18]
          vmovntps  %xmm11, (%rax)

// CHECK: vldmxcsr  -4(%rip)
// CHECK: encoding: [0xc5,0xf8,0xae,0x15,0xfc,0xff,0xff,0xff]
          vldmxcsr  -4(%rip)

// CHECK: vstmxcsr  -4(%rsp)
// CHECK: encoding: [0xc5,0xf8,0xae,0x5c,0x24,0xfc]
          vstmxcsr  -4(%rsp)

// CHECK: vpsubb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xf8,0xeb]
          vpsubb  %xmm11, %xmm12, %xmm13

// CHECK: vpsubb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xf8,0x28]
          vpsubb  (%rax), %xmm12, %xmm13

// CHECK: vpsubw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xf9,0xeb]
          vpsubw  %xmm11, %xmm12, %xmm13

// CHECK: vpsubw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xf9,0x28]
          vpsubw  (%rax), %xmm12, %xmm13

// CHECK: vpsubd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xfa,0xeb]
          vpsubd  %xmm11, %xmm12, %xmm13

// CHECK: vpsubd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xfa,0x28]
          vpsubd  (%rax), %xmm12, %xmm13

// CHECK: vpsubq  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xfb,0xeb]
          vpsubq  %xmm11, %xmm12, %xmm13

// CHECK: vpsubq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xfb,0x28]
          vpsubq  (%rax), %xmm12, %xmm13

// CHECK: vpsubsb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xe8,0xeb]
          vpsubsb  %xmm11, %xmm12, %xmm13

// CHECK: vpsubsb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xe8,0x28]
          vpsubsb  (%rax), %xmm12, %xmm13

// CHECK: vpsubsw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xe9,0xeb]
          vpsubsw  %xmm11, %xmm12, %xmm13

// CHECK: vpsubsw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xe9,0x28]
          vpsubsw  (%rax), %xmm12, %xmm13

// CHECK: vpsubusb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xd8,0xeb]
          vpsubusb  %xmm11, %xmm12, %xmm13

// CHECK: vpsubusb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xd8,0x28]
          vpsubusb  (%rax), %xmm12, %xmm13

// CHECK: vpsubusw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xd9,0xeb]
          vpsubusw  %xmm11, %xmm12, %xmm13

// CHECK: vpsubusw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xd9,0x28]
          vpsubusw  (%rax), %xmm12, %xmm13

// CHECK: vpaddb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xfc,0xeb]
          vpaddb  %xmm11, %xmm12, %xmm13

// CHECK: vpaddb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xfc,0x28]
          vpaddb  (%rax), %xmm12, %xmm13

// CHECK: vpaddw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xfd,0xeb]
          vpaddw  %xmm11, %xmm12, %xmm13

// CHECK: vpaddw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xfd,0x28]
          vpaddw  (%rax), %xmm12, %xmm13

// CHECK: vpaddd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xfe,0xeb]
          vpaddd  %xmm11, %xmm12, %xmm13

// CHECK: vpaddd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xfe,0x28]
          vpaddd  (%rax), %xmm12, %xmm13

// CHECK: vpaddq  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xd4,0xeb]
          vpaddq  %xmm11, %xmm12, %xmm13

// CHECK: vpaddq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xd4,0x28]
          vpaddq  (%rax), %xmm12, %xmm13

// CHECK: vpaddsb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xec,0xeb]
          vpaddsb  %xmm11, %xmm12, %xmm13

// CHECK: vpaddsb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xec,0x28]
          vpaddsb  (%rax), %xmm12, %xmm13

// CHECK: vpaddsw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xed,0xeb]
          vpaddsw  %xmm11, %xmm12, %xmm13

// CHECK: vpaddsw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xed,0x28]
          vpaddsw  (%rax), %xmm12, %xmm13

// CHECK: vpaddusb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xdc,0xeb]
          vpaddusb  %xmm11, %xmm12, %xmm13

// CHECK: vpaddusb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xdc,0x28]
          vpaddusb  (%rax), %xmm12, %xmm13

// CHECK: vpaddusw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xdd,0xeb]
          vpaddusw  %xmm11, %xmm12, %xmm13

// CHECK: vpaddusw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xdd,0x28]
          vpaddusw  (%rax), %xmm12, %xmm13

// CHECK: vpmulhuw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xe4,0xeb]
          vpmulhuw  %xmm11, %xmm12, %xmm13

// CHECK: vpmulhuw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xe4,0x28]
          vpmulhuw  (%rax), %xmm12, %xmm13

// CHECK: vpmulhw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xe5,0xeb]
          vpmulhw  %xmm11, %xmm12, %xmm13

// CHECK: vpmulhw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xe5,0x28]
          vpmulhw  (%rax), %xmm12, %xmm13

// CHECK: vpmullw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xd5,0xeb]
          vpmullw  %xmm11, %xmm12, %xmm13

// CHECK: vpmullw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xd5,0x28]
          vpmullw  (%rax), %xmm12, %xmm13

// CHECK: vpmuludq  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xf4,0xeb]
          vpmuludq  %xmm11, %xmm12, %xmm13

// CHECK: vpmuludq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xf4,0x28]
          vpmuludq  (%rax), %xmm12, %xmm13

// CHECK: vpavgb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xe0,0xeb]
          vpavgb  %xmm11, %xmm12, %xmm13

// CHECK: vpavgb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xe0,0x28]
          vpavgb  (%rax), %xmm12, %xmm13

// CHECK: vpavgw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xe3,0xeb]
          vpavgw  %xmm11, %xmm12, %xmm13

// CHECK: vpavgw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xe3,0x28]
          vpavgw  (%rax), %xmm12, %xmm13

// CHECK: vpminsw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xea,0xeb]
          vpminsw  %xmm11, %xmm12, %xmm13

// CHECK: vpminsw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xea,0x28]
          vpminsw  (%rax), %xmm12, %xmm13

// CHECK: vpminub  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xda,0xeb]
          vpminub  %xmm11, %xmm12, %xmm13

// CHECK: vpminub  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xda,0x28]
          vpminub  (%rax), %xmm12, %xmm13

// CHECK: vpmaxsw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xee,0xeb]
          vpmaxsw  %xmm11, %xmm12, %xmm13

// CHECK: vpmaxsw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xee,0x28]
          vpmaxsw  (%rax), %xmm12, %xmm13

// CHECK: vpmaxub  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xde,0xeb]
          vpmaxub  %xmm11, %xmm12, %xmm13

// CHECK: vpmaxub  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xde,0x28]
          vpmaxub  (%rax), %xmm12, %xmm13

// CHECK: vpsadbw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xf6,0xeb]
          vpsadbw  %xmm11, %xmm12, %xmm13

// CHECK: vpsadbw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xf6,0x28]
          vpsadbw  (%rax), %xmm12, %xmm13

// CHECK: vpsllw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xf1,0xeb]
          vpsllw  %xmm11, %xmm12, %xmm13

// CHECK: vpsllw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xf1,0x28]
          vpsllw  (%rax), %xmm12, %xmm13

// CHECK: vpslld  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xf2,0xeb]
          vpslld  %xmm11, %xmm12, %xmm13

// CHECK: vpslld  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xf2,0x28]
          vpslld  (%rax), %xmm12, %xmm13

// CHECK: vpsllq  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xf3,0xeb]
          vpsllq  %xmm11, %xmm12, %xmm13

// CHECK: vpsllq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xf3,0x28]
          vpsllq  (%rax), %xmm12, %xmm13

// CHECK: vpsraw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xe1,0xeb]
          vpsraw  %xmm11, %xmm12, %xmm13

// CHECK: vpsraw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xe1,0x28]
          vpsraw  (%rax), %xmm12, %xmm13

// CHECK: vpsrad  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xe2,0xeb]
          vpsrad  %xmm11, %xmm12, %xmm13

// CHECK: vpsrad  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xe2,0x28]
          vpsrad  (%rax), %xmm12, %xmm13

// CHECK: vpsrlw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xd1,0xeb]
          vpsrlw  %xmm11, %xmm12, %xmm13

// CHECK: vpsrlw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xd1,0x28]
          vpsrlw  (%rax), %xmm12, %xmm13

// CHECK: vpsrld  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xd2,0xeb]
          vpsrld  %xmm11, %xmm12, %xmm13

// CHECK: vpsrld  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xd2,0x28]
          vpsrld  (%rax), %xmm12, %xmm13

// CHECK: vpsrlq  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xd3,0xeb]
          vpsrlq  %xmm11, %xmm12, %xmm13

// CHECK: vpsrlq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xd3,0x28]
          vpsrlq  (%rax), %xmm12, %xmm13

// CHECK: vpslld  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x72,0xf4,0x0a]
          vpslld  $10, %xmm12, %xmm13

// CHECK: vpslldq  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x73,0xfc,0x0a]
          vpslldq  $10, %xmm12, %xmm13

// CHECK: vpsllq  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x73,0xf4,0x0a]
          vpsllq  $10, %xmm12, %xmm13

// CHECK: vpsllw  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x71,0xf4,0x0a]
          vpsllw  $10, %xmm12, %xmm13

// CHECK: vpsrad  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x72,0xe4,0x0a]
          vpsrad  $10, %xmm12, %xmm13

// CHECK: vpsraw  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x71,0xe4,0x0a]
          vpsraw  $10, %xmm12, %xmm13

// CHECK: vpsrld  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x72,0xd4,0x0a]
          vpsrld  $10, %xmm12, %xmm13

// CHECK: vpsrldq  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x73,0xdc,0x0a]
          vpsrldq  $10, %xmm12, %xmm13

// CHECK: vpsrlq  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x73,0xd4,0x0a]
          vpsrlq  $10, %xmm12, %xmm13

// CHECK: vpsrlw  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x71,0xd4,0x0a]
          vpsrlw  $10, %xmm12, %xmm13

// CHECK: vpslld  $10, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0xc1,0x11,0x72,0xf4,0x0a]
          vpslld  $10, %xmm12, %xmm13

// CHECK: vpand  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xdb,0xeb]
          vpand  %xmm11, %xmm12, %xmm13

// CHECK: vpand  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xdb,0x28]
          vpand  (%rax), %xmm12, %xmm13

// CHECK: vpor  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xeb,0xeb]
          vpor  %xmm11, %xmm12, %xmm13

// CHECK: vpor  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xeb,0x28]
          vpor  (%rax), %xmm12, %xmm13

// CHECK: vpxor  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xef,0xeb]
          vpxor  %xmm11, %xmm12, %xmm13

// CHECK: vpxor  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xef,0x28]
          vpxor  (%rax), %xmm12, %xmm13

// CHECK: vpandn  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xdf,0xeb]
          vpandn  %xmm11, %xmm12, %xmm13

// CHECK: vpandn  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xdf,0x28]
          vpandn  (%rax), %xmm12, %xmm13

// CHECK: vpcmpeqb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x74,0xeb]
          vpcmpeqb  %xmm11, %xmm12, %xmm13

// CHECK: vpcmpeqb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x74,0x28]
          vpcmpeqb  (%rax), %xmm12, %xmm13

// CHECK: vpcmpeqw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x75,0xeb]
          vpcmpeqw  %xmm11, %xmm12, %xmm13

// CHECK: vpcmpeqw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x75,0x28]
          vpcmpeqw  (%rax), %xmm12, %xmm13

// CHECK: vpcmpeqd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x76,0xeb]
          vpcmpeqd  %xmm11, %xmm12, %xmm13

// CHECK: vpcmpeqd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x76,0x28]
          vpcmpeqd  (%rax), %xmm12, %xmm13

// CHECK: vpcmpgtb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x64,0xeb]
          vpcmpgtb  %xmm11, %xmm12, %xmm13

// CHECK: vpcmpgtb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x64,0x28]
          vpcmpgtb  (%rax), %xmm12, %xmm13

// CHECK: vpcmpgtw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x65,0xeb]
          vpcmpgtw  %xmm11, %xmm12, %xmm13

// CHECK: vpcmpgtw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x65,0x28]
          vpcmpgtw  (%rax), %xmm12, %xmm13

// CHECK: vpcmpgtd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x66,0xeb]
          vpcmpgtd  %xmm11, %xmm12, %xmm13

// CHECK: vpcmpgtd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x66,0x28]
          vpcmpgtd  (%rax), %xmm12, %xmm13

// CHECK: vpacksswb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x63,0xeb]
          vpacksswb  %xmm11, %xmm12, %xmm13

// CHECK: vpacksswb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x63,0x28]
          vpacksswb  (%rax), %xmm12, %xmm13

// CHECK: vpackssdw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x6b,0xeb]
          vpackssdw  %xmm11, %xmm12, %xmm13

// CHECK: vpackssdw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x6b,0x28]
          vpackssdw  (%rax), %xmm12, %xmm13

// CHECK: vpackuswb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x67,0xeb]
          vpackuswb  %xmm11, %xmm12, %xmm13

// CHECK: vpackuswb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x67,0x28]
          vpackuswb  (%rax), %xmm12, %xmm13

// CHECK: vpshufd  $4, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x79,0x70,0xec,0x04]
          vpshufd  $4, %xmm12, %xmm13

// CHECK: vpshufd  $4, (%rax), %xmm13
// CHECK: encoding: [0xc5,0x79,0x70,0x28,0x04]
          vpshufd  $4, (%rax), %xmm13

// CHECK: vpshufhw  $4, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x7a,0x70,0xec,0x04]
          vpshufhw  $4, %xmm12, %xmm13

// CHECK: vpshufhw  $4, (%rax), %xmm13
// CHECK: encoding: [0xc5,0x7a,0x70,0x28,0x04]
          vpshufhw  $4, (%rax), %xmm13

// CHECK: vpshuflw  $4, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x7b,0x70,0xec,0x04]
          vpshuflw  $4, %xmm12, %xmm13

// CHECK: vpshuflw  $4, (%rax), %xmm13
// CHECK: encoding: [0xc5,0x7b,0x70,0x28,0x04]
          vpshuflw  $4, (%rax), %xmm13

// CHECK: vpunpcklbw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x60,0xeb]
          vpunpcklbw  %xmm11, %xmm12, %xmm13

// CHECK: vpunpcklbw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x60,0x28]
          vpunpcklbw  (%rax), %xmm12, %xmm13

// CHECK: vpunpcklwd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x61,0xeb]
          vpunpcklwd  %xmm11, %xmm12, %xmm13

// CHECK: vpunpcklwd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x61,0x28]
          vpunpcklwd  (%rax), %xmm12, %xmm13

// CHECK: vpunpckldq  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x62,0xeb]
          vpunpckldq  %xmm11, %xmm12, %xmm13

// CHECK: vpunpckldq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x62,0x28]
          vpunpckldq  (%rax), %xmm12, %xmm13

// CHECK: vpunpcklqdq  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x6c,0xeb]
          vpunpcklqdq  %xmm11, %xmm12, %xmm13

// CHECK: vpunpcklqdq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x6c,0x28]
          vpunpcklqdq  (%rax), %xmm12, %xmm13

// CHECK: vpunpckhbw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x68,0xeb]
          vpunpckhbw  %xmm11, %xmm12, %xmm13

// CHECK: vpunpckhbw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x68,0x28]
          vpunpckhbw  (%rax), %xmm12, %xmm13

// CHECK: vpunpckhwd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x69,0xeb]
          vpunpckhwd  %xmm11, %xmm12, %xmm13

// CHECK: vpunpckhwd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x69,0x28]
          vpunpckhwd  (%rax), %xmm12, %xmm13

// CHECK: vpunpckhdq  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x6a,0xeb]
          vpunpckhdq  %xmm11, %xmm12, %xmm13

// CHECK: vpunpckhdq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x6a,0x28]
          vpunpckhdq  (%rax), %xmm12, %xmm13

// CHECK: vpunpckhqdq  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x6d,0xeb]
          vpunpckhqdq  %xmm11, %xmm12, %xmm13

// CHECK: vpunpckhqdq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x6d,0x28]
          vpunpckhqdq  (%rax), %xmm12, %xmm13

// CHECK: vpinsrw  $7, %eax, %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc4,0xe8,0x07]
          vpinsrw  $7, %eax, %xmm12, %xmm13

// CHECK: vpinsrw  $7, (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0xc4,0x28,0x07]
          vpinsrw  $7, (%rax), %xmm12, %xmm13

// CHECK: vpextrw  $7, %xmm12, %eax
// CHECK: encoding: [0xc4,0xc1,0x79,0xc5,0xc4,0x07]
          vpextrw  $7, %xmm12, %eax

// CHECK: vpmovmskb  %xmm12, %eax
// CHECK: encoding: [0xc4,0xc1,0x79,0xd7,0xc4]
          vpmovmskb  %xmm12, %eax

// CHECK: vmaskmovdqu  %xmm14, %xmm15
// CHECK: encoding: [0xc4,0x41,0x79,0xf7,0xfe]
          vmaskmovdqu  %xmm14, %xmm15

// CHECK: vmovd  %eax, %xmm14
// CHECK: encoding: [0xc5,0x79,0x6e,0xf0]
          vmovd  %eax, %xmm14

// CHECK: vmovd  (%rax), %xmm14
// CHECK: encoding: [0xc5,0x79,0x6e,0x30]
          vmovd  (%rax), %xmm14

// CHECK: vmovd  %xmm14, (%rax)
// CHECK: encoding: [0xc5,0x79,0x7e,0x30]
          vmovd  %xmm14, (%rax)

// CHECK: vmovq  %rax, %xmm14
// CHECK: encoding: [0xc4,0x61,0xf9,0x6e,0xf0]
          vmovd  %rax, %xmm14

// CHECK: vmovq %xmm0, %rax
// CHECK: encoding: [0xc4,0xe1,0xf9,0x7e,0xc0]
          vmovd %xmm0, %rax

// CHECK: vmovq  %xmm14, (%rax)
// CHECK: encoding: [0xc5,0x79,0xd6,0x30]
          vmovq  %xmm14, (%rax)

// CHECK: vmovq  %xmm14, %xmm12
// CHECK: encoding: [0xc4,0x41,0x7a,0x7e,0xe6]
          vmovq  %xmm14, %xmm12

// CHECK: vmovq  (%rax), %xmm14
// CHECK: encoding: [0xc5,0x7a,0x7e,0x30]
          vmovq  (%rax), %xmm14

// CHECK: vmovq  %rax, %xmm14
// CHECK: encoding: [0xc4,0x61,0xf9,0x6e,0xf0]
          vmovq  %rax, %xmm14

// CHECK: vmovq  %xmm14, %rax
// CHECK: encoding: [0xc4,0x61,0xf9,0x7e,0xf0]
          vmovq  %xmm14, %rax

// CHECK: vcvtpd2dq  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x7b,0xe6,0xe3]
          vcvtpd2dq  %xmm11, %xmm12

// CHECK: vcvtdq2pd  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x7a,0xe6,0xe3]
          vcvtdq2pd  %xmm11, %xmm12

// CHECK: vcvtdq2pd  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x7a,0xe6,0x20]
          vcvtdq2pd  (%rax), %xmm12

// CHECK: vmovshdup  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x7a,0x16,0xe3]
          vmovshdup  %xmm11, %xmm12

// CHECK: vmovshdup  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x7a,0x16,0x20]
          vmovshdup  (%rax), %xmm12

// CHECK: vmovsldup  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x7a,0x12,0xe3]
          vmovsldup  %xmm11, %xmm12

// CHECK: vmovsldup  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x7a,0x12,0x20]
          vmovsldup  (%rax), %xmm12

// CHECK: vmovddup  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x41,0x7b,0x12,0xe3]
          vmovddup  %xmm11, %xmm12

// CHECK: vmovddup  (%rax), %xmm12
// CHECK: encoding: [0xc5,0x7b,0x12,0x20]
          vmovddup  (%rax), %xmm12

// CHECK: vaddsubps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0xd0,0xeb]
          vaddsubps  %xmm11, %xmm12, %xmm13

// CHECK: vaddsubps  (%rax), %xmm11, %xmm12
// CHECK: encoding: [0xc5,0x23,0xd0,0x20]
          vaddsubps  (%rax), %xmm11, %xmm12

// CHECK: vaddsubpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0xd0,0xeb]
          vaddsubpd  %xmm11, %xmm12, %xmm13

// CHECK: vaddsubpd  (%rax), %xmm11, %xmm12
// CHECK: encoding: [0xc5,0x21,0xd0,0x20]
          vaddsubpd  (%rax), %xmm11, %xmm12

// CHECK: vhaddps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0x7c,0xeb]
          vhaddps  %xmm11, %xmm12, %xmm13

// CHECK: vhaddps  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0x7c,0x28]
          vhaddps  (%rax), %xmm12, %xmm13

// CHECK: vhaddpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x7c,0xeb]
          vhaddpd  %xmm11, %xmm12, %xmm13

// CHECK: vhaddpd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x7c,0x28]
          vhaddpd  (%rax), %xmm12, %xmm13

// CHECK: vhsubps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x1b,0x7d,0xeb]
          vhsubps  %xmm11, %xmm12, %xmm13

// CHECK: vhsubps  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x1b,0x7d,0x28]
          vhsubps  (%rax), %xmm12, %xmm13

// CHECK: vhsubpd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x19,0x7d,0xeb]
          vhsubpd  %xmm11, %xmm12, %xmm13

// CHECK: vhsubpd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc5,0x19,0x7d,0x28]
          vhsubpd  (%rax), %xmm12, %xmm13

// CHECK: vpabsb  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x42,0x79,0x1c,0xe3]
          vpabsb  %xmm11, %xmm12

// CHECK: vpabsb  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x1c,0x20]
          vpabsb  (%rax), %xmm12

// CHECK: vpabsw  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x42,0x79,0x1d,0xe3]
          vpabsw  %xmm11, %xmm12

// CHECK: vpabsw  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x1d,0x20]
          vpabsw  (%rax), %xmm12

// CHECK: vpabsd  %xmm11, %xmm12
// CHECK: encoding: [0xc4,0x42,0x79,0x1e,0xe3]
          vpabsd  %xmm11, %xmm12

// CHECK: vpabsd  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x1e,0x20]
          vpabsd  (%rax), %xmm12

// CHECK: vphaddw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x01,0xeb]
          vphaddw  %xmm11, %xmm12, %xmm13

// CHECK: vphaddw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x01,0x28]
          vphaddw  (%rax), %xmm12, %xmm13

// CHECK: vphaddd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x02,0xeb]
          vphaddd  %xmm11, %xmm12, %xmm13

// CHECK: vphaddd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x02,0x28]
          vphaddd  (%rax), %xmm12, %xmm13

// CHECK: vphaddsw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x03,0xeb]
          vphaddsw  %xmm11, %xmm12, %xmm13

// CHECK: vphaddsw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x03,0x28]
          vphaddsw  (%rax), %xmm12, %xmm13

// CHECK: vphsubw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x05,0xeb]
          vphsubw  %xmm11, %xmm12, %xmm13

// CHECK: vphsubw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x05,0x28]
          vphsubw  (%rax), %xmm12, %xmm13

// CHECK: vphsubd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x06,0xeb]
          vphsubd  %xmm11, %xmm12, %xmm13

// CHECK: vphsubd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x06,0x28]
          vphsubd  (%rax), %xmm12, %xmm13

// CHECK: vphsubsw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x07,0xeb]
          vphsubsw  %xmm11, %xmm12, %xmm13

// CHECK: vphsubsw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x07,0x28]
          vphsubsw  (%rax), %xmm12, %xmm13

// CHECK: vpmaddubsw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x04,0xeb]
          vpmaddubsw  %xmm11, %xmm12, %xmm13

// CHECK: vpmaddubsw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x04,0x28]
          vpmaddubsw  (%rax), %xmm12, %xmm13

// CHECK: vpshufb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x00,0xeb]
          vpshufb  %xmm11, %xmm12, %xmm13

// CHECK: vpshufb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x00,0x28]
          vpshufb  (%rax), %xmm12, %xmm13

// CHECK: vpsignb  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x08,0xeb]
          vpsignb  %xmm11, %xmm12, %xmm13

// CHECK: vpsignb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x08,0x28]
          vpsignb  (%rax), %xmm12, %xmm13

// CHECK: vpsignw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x09,0xeb]
          vpsignw  %xmm11, %xmm12, %xmm13

// CHECK: vpsignw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x09,0x28]
          vpsignw  (%rax), %xmm12, %xmm13

// CHECK: vpsignd  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x0a,0xeb]
          vpsignd  %xmm11, %xmm12, %xmm13

// CHECK: vpsignd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x0a,0x28]
          vpsignd  (%rax), %xmm12, %xmm13

// CHECK: vpmulhrsw  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x19,0x0b,0xeb]
          vpmulhrsw  %xmm11, %xmm12, %xmm13

// CHECK: vpmulhrsw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x0b,0x28]
          vpmulhrsw  (%rax), %xmm12, %xmm13

// CHECK: vpalignr  $7, %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x43,0x19,0x0f,0xeb,0x07]
          vpalignr  $7, %xmm11, %xmm12, %xmm13

// CHECK: vpalignr  $7, (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x63,0x19,0x0f,0x28,0x07]
          vpalignr  $7, (%rax), %xmm12, %xmm13

// CHECK: vroundsd  $7, %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x43,0x19,0x0b,0xeb,0x07]
          vroundsd  $7, %xmm11, %xmm12, %xmm13

// CHECK: vroundsd  $7, (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x63,0x19,0x0b,0x28,0x07]
          vroundsd  $7, (%rax), %xmm12, %xmm13

// CHECK: vroundss  $7, %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x43,0x19,0x0a,0xeb,0x07]
          vroundss  $7, %xmm11, %xmm12, %xmm13

// CHECK: vroundss  $7, (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x63,0x19,0x0a,0x28,0x07]
          vroundss  $7, (%rax), %xmm12, %xmm13

// CHECK: vroundpd  $7, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x43,0x79,0x09,0xec,0x07]
          vroundpd  $7, %xmm12, %xmm13

// CHECK: vroundpd  $7, (%rax), %xmm13
// CHECK: encoding: [0xc4,0x63,0x79,0x09,0x28,0x07]
          vroundpd  $7, (%rax), %xmm13

// CHECK: vroundps  $7, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x43,0x79,0x08,0xec,0x07]
          vroundps  $7, %xmm12, %xmm13

// CHECK: vroundps  $7, (%rax), %xmm13
// CHECK: encoding: [0xc4,0x63,0x79,0x08,0x28,0x07]
          vroundps  $7, (%rax), %xmm13

// CHECK: vphminposuw  %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x42,0x79,0x41,0xec]
          vphminposuw  %xmm12, %xmm13

// CHECK: vphminposuw  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x41,0x20]
          vphminposuw  (%rax), %xmm12

// CHECK: vpackusdw  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x2b,0xdc]
          vpackusdw  %xmm12, %xmm13, %xmm11

// CHECK: vpackusdw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x2b,0x28]
          vpackusdw  (%rax), %xmm12, %xmm13

// CHECK: vpcmpeqq  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x29,0xdc]
          vpcmpeqq  %xmm12, %xmm13, %xmm11

// CHECK: vpcmpeqq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x29,0x28]
          vpcmpeqq  (%rax), %xmm12, %xmm13

// CHECK: vpminsb  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x38,0xdc]
          vpminsb  %xmm12, %xmm13, %xmm11

// CHECK: vpminsb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x38,0x28]
          vpminsb  (%rax), %xmm12, %xmm13

// CHECK: vpminsd  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x39,0xdc]
          vpminsd  %xmm12, %xmm13, %xmm11

// CHECK: vpminsd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x39,0x28]
          vpminsd  (%rax), %xmm12, %xmm13

// CHECK: vpminud  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x3b,0xdc]
          vpminud  %xmm12, %xmm13, %xmm11

// CHECK: vpminud  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x3b,0x28]
          vpminud  (%rax), %xmm12, %xmm13

// CHECK: vpminuw  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x3a,0xdc]
          vpminuw  %xmm12, %xmm13, %xmm11

// CHECK: vpminuw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x3a,0x28]
          vpminuw  (%rax), %xmm12, %xmm13

// CHECK: vpmaxsb  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x3c,0xdc]
          vpmaxsb  %xmm12, %xmm13, %xmm11

// CHECK: vpmaxsb  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x3c,0x28]
          vpmaxsb  (%rax), %xmm12, %xmm13

// CHECK: vpmaxsd  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x3d,0xdc]
          vpmaxsd  %xmm12, %xmm13, %xmm11

// CHECK: vpmaxsd  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x3d,0x28]
          vpmaxsd  (%rax), %xmm12, %xmm13

// CHECK: vpmaxud  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x3f,0xdc]
          vpmaxud  %xmm12, %xmm13, %xmm11

// CHECK: vpmaxud  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x3f,0x28]
          vpmaxud  (%rax), %xmm12, %xmm13

// CHECK: vpmaxuw  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x3e,0xdc]
          vpmaxuw  %xmm12, %xmm13, %xmm11

// CHECK: vpmaxuw  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x3e,0x28]
          vpmaxuw  (%rax), %xmm12, %xmm13

// CHECK: vpmuldq  %xmm12, %xmm13, %xmm11
// CHECK: encoding: [0xc4,0x42,0x11,0x28,0xdc]
          vpmuldq  %xmm12, %xmm13, %xmm11

// CHECK: vpmuldq  (%rax), %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x62,0x19,0x28,0x28]
          vpmuldq  (%rax), %xmm12, %xmm13

// CHECK: vpmulld  %xmm12, %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x42,0x51,0x40,0xdc]
          vpmulld  %xmm12, %xmm5, %xmm11

// CHECK: vpmulld  (%rax), %xmm5, %xmm13
// CHECK: encoding: [0xc4,0x62,0x51,0x40,0x28]
          vpmulld  (%rax), %xmm5, %xmm13

// CHECK: vblendps  $3, %xmm12, %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x43,0x51,0x0c,0xdc,0x03]
          vblendps  $3, %xmm12, %xmm5, %xmm11

// CHECK: vblendps  $3, (%rax), %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x63,0x51,0x0c,0x18,0x03]
          vblendps  $3, (%rax), %xmm5, %xmm11

// CHECK: vblendpd  $3, %xmm12, %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x43,0x51,0x0d,0xdc,0x03]
          vblendpd  $3, %xmm12, %xmm5, %xmm11

// CHECK: vblendpd  $3, (%rax), %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x63,0x51,0x0d,0x18,0x03]
          vblendpd  $3, (%rax), %xmm5, %xmm11

// CHECK: vpblendw  $3, %xmm12, %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x43,0x51,0x0e,0xdc,0x03]
          vpblendw  $3, %xmm12, %xmm5, %xmm11

// CHECK: vpblendw  $3, (%rax), %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x63,0x51,0x0e,0x18,0x03]
          vpblendw  $3, (%rax), %xmm5, %xmm11

// CHECK: vmpsadbw  $3, %xmm12, %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x43,0x51,0x42,0xdc,0x03]
          vmpsadbw  $3, %xmm12, %xmm5, %xmm11

// CHECK: vmpsadbw  $3, (%rax), %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x63,0x51,0x42,0x18,0x03]
          vmpsadbw  $3, (%rax), %xmm5, %xmm11

// CHECK: vdpps  $3, %xmm12, %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x43,0x51,0x40,0xdc,0x03]
          vdpps  $3, %xmm12, %xmm5, %xmm11

// CHECK: vdpps  $3, (%rax), %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x63,0x51,0x40,0x18,0x03]
          vdpps  $3, (%rax), %xmm5, %xmm11

// CHECK: vdppd  $3, %xmm12, %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x43,0x51,0x41,0xdc,0x03]
          vdppd  $3, %xmm12, %xmm5, %xmm11

// CHECK: vdppd  $3, (%rax), %xmm5, %xmm11
// CHECK: encoding: [0xc4,0x63,0x51,0x41,0x18,0x03]
          vdppd  $3, (%rax), %xmm5, %xmm11

// CHECK: vblendvpd  %xmm12, %xmm5, %xmm11, %xmm13
// CHECK: encoding: [0xc4,0x63,0x21,0x4b,0xed,0xc0]
          vblendvpd  %xmm12, %xmm5, %xmm11, %xmm13

// CHECK: vblendvpd  %xmm12, (%rax), %xmm11, %xmm13
// CHECK: encoding: [0xc4,0x63,0x21,0x4b,0x28,0xc0]
          vblendvpd  %xmm12, (%rax), %xmm11, %xmm13

// CHECK: vblendvps  %xmm12, %xmm5, %xmm11, %xmm13
// CHECK: encoding: [0xc4,0x63,0x21,0x4a,0xed,0xc0]
          vblendvps  %xmm12, %xmm5, %xmm11, %xmm13

// CHECK: vblendvps  %xmm12, (%rax), %xmm11, %xmm13
// CHECK: encoding: [0xc4,0x63,0x21,0x4a,0x28,0xc0]
          vblendvps  %xmm12, (%rax), %xmm11, %xmm13

// CHECK: vpblendvb  %xmm12, %xmm5, %xmm11, %xmm13
// CHECK: encoding: [0xc4,0x63,0x21,0x4c,0xed,0xc0]
          vpblendvb  %xmm12, %xmm5, %xmm11, %xmm13

// CHECK: vpblendvb  %xmm12, (%rax), %xmm11, %xmm13
// CHECK: encoding: [0xc4,0x63,0x21,0x4c,0x28,0xc0]
          vpblendvb  %xmm12, (%rax), %xmm11, %xmm13

// CHECK: vpmovsxbw  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x20,0xd4]
          vpmovsxbw  %xmm12, %xmm10

// CHECK: vpmovsxbw  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x20,0x20]
          vpmovsxbw  (%rax), %xmm12

// CHECK: vpmovsxwd  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x23,0xd4]
          vpmovsxwd  %xmm12, %xmm10

// CHECK: vpmovsxwd  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x23,0x20]
          vpmovsxwd  (%rax), %xmm12

// CHECK: vpmovsxdq  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x25,0xd4]
          vpmovsxdq  %xmm12, %xmm10

// CHECK: vpmovsxdq  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x25,0x20]
          vpmovsxdq  (%rax), %xmm12

// CHECK: vpmovzxbw  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x30,0xd4]
          vpmovzxbw  %xmm12, %xmm10

// CHECK: vpmovzxbw  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x30,0x20]
          vpmovzxbw  (%rax), %xmm12

// CHECK: vpmovzxwd  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x33,0xd4]
          vpmovzxwd  %xmm12, %xmm10

// CHECK: vpmovzxwd  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x33,0x20]
          vpmovzxwd  (%rax), %xmm12

// CHECK: vpmovzxdq  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x35,0xd4]
          vpmovzxdq  %xmm12, %xmm10

// CHECK: vpmovzxdq  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x35,0x20]
          vpmovzxdq  (%rax), %xmm12

// CHECK: vpmovsxbq  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x22,0xd4]
          vpmovsxbq  %xmm12, %xmm10

// CHECK: vpmovsxbq  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x22,0x20]
          vpmovsxbq  (%rax), %xmm12

// CHECK: vpmovzxbq  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x32,0xd4]
          vpmovzxbq  %xmm12, %xmm10

// CHECK: vpmovzxbq  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x32,0x20]
          vpmovzxbq  (%rax), %xmm12

// CHECK: vpmovsxbd  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x21,0xd4]
          vpmovsxbd  %xmm12, %xmm10

// CHECK: vpmovsxbd  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x21,0x20]
          vpmovsxbd  (%rax), %xmm12

// CHECK: vpmovsxwq  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x24,0xd4]
          vpmovsxwq  %xmm12, %xmm10

// CHECK: vpmovsxwq  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x24,0x20]
          vpmovsxwq  (%rax), %xmm12

// CHECK: vpmovzxbd  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x31,0xd4]
          vpmovzxbd  %xmm12, %xmm10

// CHECK: vpmovzxbd  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x31,0x20]
          vpmovzxbd  (%rax), %xmm12

// CHECK: vpmovzxwq  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x34,0xd4]
          vpmovzxwq  %xmm12, %xmm10

// CHECK: vpmovzxwq  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x34,0x20]
          vpmovzxwq  (%rax), %xmm12

// CHECK: vpextrw  $7, %xmm12, %eax
// CHECK: encoding: [0xc4,0xc1,0x79,0xc5,0xc4,0x07]
          vpextrw  $7, %xmm12, %eax

// CHECK: vpextrw  $7, %xmm12, (%rax)
// CHECK: encoding: [0xc4,0x63,0x79,0x15,0x20,0x07]
          vpextrw  $7, %xmm12, (%rax)

// CHECK: vpextrd  $7, %xmm12, %eax
// CHECK: encoding: [0xc4,0x63,0x79,0x16,0xe0,0x07]
          vpextrd  $7, %xmm12, %eax

// CHECK: vpextrd  $7, %xmm12, (%rax)
// CHECK: encoding: [0xc4,0x63,0x79,0x16,0x20,0x07]
          vpextrd  $7, %xmm12, (%rax)

// CHECK: vpextrb  $7, %xmm12, %eax
// CHECK: encoding: [0xc4,0x63,0x79,0x14,0xe0,0x07]
          vpextrb  $7, %xmm12, %eax

// CHECK: vpextrb  $7, %xmm12, (%rax)
// CHECK: encoding: [0xc4,0x63,0x79,0x14,0x20,0x07]
          vpextrb  $7, %xmm12, (%rax)

// CHECK: vpextrq  $7, %xmm12, %rcx
// CHECK: encoding: [0xc4,0x63,0xf9,0x16,0xe1,0x07]
          vpextrq  $7, %xmm12, %rcx

// CHECK: vpextrq  $7, %xmm12, (%rcx)
// CHECK: encoding: [0xc4,0x63,0xf9,0x16,0x21,0x07]
          vpextrq  $7, %xmm12, (%rcx)

// CHECK: vextractps  $7, %xmm12, (%rax)
// CHECK: encoding: [0xc4,0x63,0x79,0x17,0x20,0x07]
          vextractps  $7, %xmm12, (%rax)

// CHECK: vextractps  $7, %xmm12, %eax
// CHECK: encoding: [0xc4,0x63,0x79,0x17,0xe0,0x07]
          vextractps  $7, %xmm12, %eax

// CHECK: vpinsrw  $7, %eax, %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x19,0xc4,0xd0,0x07]
          vpinsrw  $7, %eax, %xmm12, %xmm10

// CHECK: vpinsrw  $7, (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc5,0x19,0xc4,0x10,0x07]
          vpinsrw  $7, (%rax), %xmm12, %xmm10

// CHECK: vpinsrb  $7, %eax, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x63,0x19,0x20,0xd0,0x07]
          vpinsrb  $7, %eax, %xmm12, %xmm10

// CHECK: vpinsrb  $7, (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x63,0x19,0x20,0x10,0x07]
          vpinsrb  $7, (%rax), %xmm12, %xmm10

// CHECK: vpinsrd  $7, %eax, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x63,0x19,0x22,0xd0,0x07]
          vpinsrd  $7, %eax, %xmm12, %xmm10

// CHECK: vpinsrd  $7, (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x63,0x19,0x22,0x10,0x07]
          vpinsrd  $7, (%rax), %xmm12, %xmm10

// CHECK: vpinsrq  $7, %rax, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x63,0x99,0x22,0xd0,0x07]
          vpinsrq  $7, %rax, %xmm12, %xmm10

// CHECK: vpinsrq  $7, (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x63,0x99,0x22,0x10,0x07]
          vpinsrq  $7, (%rax), %xmm12, %xmm10

// CHECK: vinsertps  $7, %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x43,0x29,0x21,0xdc,0x07]
          vinsertps  $7, %xmm12, %xmm10, %xmm11

// CHECK: vinsertps  $7, (%rax), %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x63,0x29,0x21,0x18,0x07]
          vinsertps  $7, (%rax), %xmm10, %xmm11

// CHECK: vptest  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x17,0xd4]
          vptest  %xmm12, %xmm10

// CHECK: vptest  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x17,0x20]
          vptest  (%rax), %xmm12

// CHECK: vmovntdqa  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x2a,0x20]
          vmovntdqa  (%rax), %xmm12

// CHECK: vpcmpgtq  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x37,0xdc]
          vpcmpgtq  %xmm12, %xmm10, %xmm11

// CHECK: vpcmpgtq  (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x62,0x29,0x37,0x28]
          vpcmpgtq  (%rax), %xmm10, %xmm13

// CHECK: vpcmpistrm  $7, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x43,0x79,0x62,0xd4,0x07]
          vpcmpistrm  $7, %xmm12, %xmm10

// CHECK: vpcmpistrm  $7, (%rax), %xmm10
// CHECK: encoding: [0xc4,0x63,0x79,0x62,0x10,0x07]
          vpcmpistrm  $7, (%rax), %xmm10

// CHECK: vpcmpestrm  $7, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x43,0x79,0x60,0xd4,0x07]
          vpcmpestrm  $7, %xmm12, %xmm10

// CHECK: vpcmpestrm  $7, (%rax), %xmm10
// CHECK: encoding: [0xc4,0x63,0x79,0x60,0x10,0x07]
          vpcmpestrm  $7, (%rax), %xmm10

// CHECK: vpcmpistri  $7, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x43,0x79,0x63,0xd4,0x07]
          vpcmpistri  $7, %xmm12, %xmm10

// CHECK: vpcmpistri  $7, (%rax), %xmm10
// CHECK: encoding: [0xc4,0x63,0x79,0x63,0x10,0x07]
          vpcmpistri  $7, (%rax), %xmm10

// CHECK: vpcmpestri  $7, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x43,0x79,0x61,0xd4,0x07]
          vpcmpestri  $7, %xmm12, %xmm10

// CHECK: vpcmpestri  $7, (%rax), %xmm10
// CHECK: encoding: [0xc4,0x63,0x79,0x61,0x10,0x07]
          vpcmpestri  $7, (%rax), %xmm10

// CHECK: vaesimc  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0xdb,0xd4]
          vaesimc  %xmm12, %xmm10

// CHECK: vaesimc  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0xdb,0x20]
          vaesimc  (%rax), %xmm12

// CHECK: vaesenc  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xdc,0xdc]
          vaesenc  %xmm12, %xmm10, %xmm11

// CHECK: vaesenc  (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x62,0x29,0xdc,0x28]
          vaesenc  (%rax), %xmm10, %xmm13

// CHECK: vaesenclast  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xdd,0xdc]
          vaesenclast  %xmm12, %xmm10, %xmm11

// CHECK: vaesenclast  (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x62,0x29,0xdd,0x28]
          vaesenclast  (%rax), %xmm10, %xmm13

// CHECK: vaesdec  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xde,0xdc]
          vaesdec  %xmm12, %xmm10, %xmm11

// CHECK: vaesdec  (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x62,0x29,0xde,0x28]
          vaesdec  (%rax), %xmm10, %xmm13

// CHECK: vaesdeclast  %xmm12, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0xdf,0xdc]
          vaesdeclast  %xmm12, %xmm10, %xmm11

// CHECK: vaesdeclast  (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x62,0x29,0xdf,0x28]
          vaesdeclast  (%rax), %xmm10, %xmm13

// CHECK: vaeskeygenassist  $7, %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x43,0x79,0xdf,0xd4,0x07]
          vaeskeygenassist  $7, %xmm12, %xmm10

// CHECK: vaeskeygenassist  $7, (%rax), %xmm10
// CHECK: encoding: [0xc4,0x63,0x79,0xdf,0x10,0x07]
          vaeskeygenassist  $7, (%rax), %xmm10

// CHECK: vcmpeq_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x08]
          vcmpeq_uqps %xmm11, %xmm12, %xmm13

// CHECK: vcmpngeps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x09]
          vcmpngeps %xmm11, %xmm12, %xmm13

// CHECK: vcmpngtps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0a]
          vcmpngtps %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalseps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0b]
          vcmpfalseps %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0c]
          vcmpneq_oqps %xmm11, %xmm12, %xmm13

// CHECK: vcmpgeps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0d]
          vcmpgeps %xmm11, %xmm12, %xmm13

// CHECK: vcmpgtps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0e]
          vcmpgtps %xmm11, %xmm12, %xmm13

// CHECK: vcmptrueps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x0f]
          vcmptrueps %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_osps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x10]
          vcmpeq_osps %xmm11, %xmm12, %xmm13

// CHECK: vcmplt_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x11]
          vcmplt_oqps %xmm11, %xmm12, %xmm13

// CHECK: vcmple_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x12]
          vcmple_oqps %xmm11, %xmm12, %xmm13

// CHECK: vcmpunord_sps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x13]
          vcmpunord_sps %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_usps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x14]
          vcmpneq_usps %xmm11, %xmm12, %xmm13

// CHECK: vcmpnlt_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x15]
          vcmpnlt_uqps %xmm11, %xmm12, %xmm13

// CHECK: vcmpnle_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x16]
          vcmpnle_uqps %xmm11, %xmm12, %xmm13

// CHECK: vcmpord_sps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x17]
          vcmpord_sps %xmm11, %xmm12, %xmm13

// CHECK: vcmpeq_usps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x18]
          vcmpeq_usps %xmm11, %xmm12, %xmm13

// CHECK: vcmpnge_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x19]
          vcmpnge_uqps %xmm11, %xmm12, %xmm13

// CHECK: vcmpngt_uqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1a]
          vcmpngt_uqps %xmm11, %xmm12, %xmm13

// CHECK: vcmpfalse_osps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1b]
          vcmpfalse_osps %xmm11, %xmm12, %xmm13

// CHECK: vcmpneq_osps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1c]
          vcmpneq_osps %xmm11, %xmm12, %xmm13

// CHECK: vcmpge_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1d]
          vcmpge_oqps %xmm11, %xmm12, %xmm13

// CHECK: vcmpgt_oqps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1e]
          vcmpgt_oqps %xmm11, %xmm12, %xmm13

// CHECK: vcmptrue_usps  %xmm11, %xmm12, %xmm13
// CHECK: encoding: [0xc4,0x41,0x18,0xc2,0xeb,0x1f]
          vcmptrue_usps %xmm11, %xmm12, %xmm13

// CHECK: vmovaps  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7c,0x28,0x20]
          vmovaps  (%rax), %ymm12

// CHECK: vmovaps  %ymm11, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7c,0x28,0xe3]
          vmovaps  %ymm11, %ymm12

// CHECK: vmovaps  %ymm11, (%rax)
// CHECK: encoding: [0xc5,0x7c,0x29,0x18]
          vmovaps  %ymm11, (%rax)

// CHECK: vmovapd  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7d,0x28,0x20]
          vmovapd  (%rax), %ymm12

// CHECK: vmovapd  %ymm11, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7d,0x28,0xe3]
          vmovapd  %ymm11, %ymm12

// CHECK: vmovapd  %ymm11, (%rax)
// CHECK: encoding: [0xc5,0x7d,0x29,0x18]
          vmovapd  %ymm11, (%rax)

// CHECK: vmovups  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7c,0x10,0x20]
          vmovups  (%rax), %ymm12

// CHECK: vmovups  %ymm11, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7c,0x10,0xe3]
          vmovups  %ymm11, %ymm12

// CHECK: vmovups  %ymm11, (%rax)
// CHECK: encoding: [0xc5,0x7c,0x11,0x18]
          vmovups  %ymm11, (%rax)

// CHECK: vmovupd  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7d,0x10,0x20]
          vmovupd  (%rax), %ymm12

// CHECK: vmovupd  %ymm11, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7d,0x10,0xe3]
          vmovupd  %ymm11, %ymm12

// CHECK: vmovupd  %ymm11, (%rax)
// CHECK: encoding: [0xc5,0x7d,0x11,0x18]
          vmovupd  %ymm11, (%rax)

// CHECK: vunpckhps  %ymm11, %ymm12, %ymm4
// CHECK: encoding: [0xc4,0xc1,0x1c,0x15,0xe3]
          vunpckhps  %ymm11, %ymm12, %ymm4

// CHECK: vunpckhpd  %ymm11, %ymm12, %ymm4
// CHECK: encoding: [0xc4,0xc1,0x1d,0x15,0xe3]
          vunpckhpd  %ymm11, %ymm12, %ymm4

// CHECK: vunpcklps  %ymm11, %ymm12, %ymm4
// CHECK: encoding: [0xc4,0xc1,0x1c,0x14,0xe3]
          vunpcklps  %ymm11, %ymm12, %ymm4

// CHECK: vunpcklpd  %ymm11, %ymm12, %ymm4
// CHECK: encoding: [0xc4,0xc1,0x1d,0x14,0xe3]
          vunpcklpd  %ymm11, %ymm12, %ymm4

// CHECK: vunpckhps  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1c,0x15,0x54,0xcb,0xfc]
          vunpckhps  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vunpckhpd  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1d,0x15,0x54,0xcb,0xfc]
          vunpckhpd  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vunpcklps  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1c,0x14,0x54,0xcb,0xfc]
          vunpcklps  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vunpcklpd  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1d,0x14,0x54,0xcb,0xfc]
          vunpcklpd  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vmovntdq  %ymm11, (%rax)
// CHECK: encoding: [0xc5,0x7d,0xe7,0x18]
          vmovntdq  %ymm11, (%rax)

// CHECK: vmovntpd  %ymm11, (%rax)
// CHECK: encoding: [0xc5,0x7d,0x2b,0x18]
          vmovntpd  %ymm11, (%rax)

// CHECK: vmovntps  %ymm11, (%rax)
// CHECK: encoding: [0xc5,0x7c,0x2b,0x18]
          vmovntps  %ymm11, (%rax)

// CHECK: vmovmskps  %xmm12, %eax
// CHECK: encoding: [0xc4,0xc1,0x78,0x50,0xc4]
          vmovmskps  %xmm12, %eax

// CHECK: vmovmskpd  %xmm12, %eax
// CHECK: encoding: [0xc4,0xc1,0x79,0x50,0xc4]
          vmovmskpd  %xmm12, %eax

// CHECK: vmaxps  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5c,0x5f,0xf4]
          vmaxps  %ymm12, %ymm4, %ymm6

// CHECK: vmaxpd  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5d,0x5f,0xf4]
          vmaxpd  %ymm12, %ymm4, %ymm6

// CHECK: vminps  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5c,0x5d,0xf4]
          vminps  %ymm12, %ymm4, %ymm6

// CHECK: vminpd  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5d,0x5d,0xf4]
          vminpd  %ymm12, %ymm4, %ymm6

// CHECK: vsubps  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5c,0x5c,0xf4]
          vsubps  %ymm12, %ymm4, %ymm6

// CHECK: vsubpd  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5d,0x5c,0xf4]
          vsubpd  %ymm12, %ymm4, %ymm6

// CHECK: vdivps  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5c,0x5e,0xf4]
          vdivps  %ymm12, %ymm4, %ymm6

// CHECK: vdivpd  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5d,0x5e,0xf4]
          vdivpd  %ymm12, %ymm4, %ymm6

// CHECK: vaddps  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5c,0x58,0xf4]
          vaddps  %ymm12, %ymm4, %ymm6

// CHECK: vaddpd  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5d,0x58,0xf4]
          vaddpd  %ymm12, %ymm4, %ymm6

// CHECK: vmulps  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5c,0x59,0xf4]
          vmulps  %ymm12, %ymm4, %ymm6

// CHECK: vmulpd  %ymm12, %ymm4, %ymm6
// CHECK: encoding: [0xc4,0xc1,0x5d,0x59,0xf4]
          vmulpd  %ymm12, %ymm4, %ymm6

// CHECK: vmaxps  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5f,0x30]
          vmaxps  (%rax), %ymm4, %ymm6

// CHECK: vmaxpd  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5f,0x30]
          vmaxpd  (%rax), %ymm4, %ymm6

// CHECK: vminps  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5d,0x30]
          vminps  (%rax), %ymm4, %ymm6

// CHECK: vminpd  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5d,0x30]
          vminpd  (%rax), %ymm4, %ymm6

// CHECK: vsubps  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5c,0x30]
          vsubps  (%rax), %ymm4, %ymm6

// CHECK: vsubpd  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5c,0x30]
          vsubpd  (%rax), %ymm4, %ymm6

// CHECK: vdivps  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x5e,0x30]
          vdivps  (%rax), %ymm4, %ymm6

// CHECK: vdivpd  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x5e,0x30]
          vdivpd  (%rax), %ymm4, %ymm6

// CHECK: vaddps  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x58,0x30]
          vaddps  (%rax), %ymm4, %ymm6

// CHECK: vaddpd  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x58,0x30]
          vaddpd  (%rax), %ymm4, %ymm6

// CHECK: vmulps  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdc,0x59,0x30]
          vmulps  (%rax), %ymm4, %ymm6

// CHECK: vmulpd  (%rax), %ymm4, %ymm6
// CHECK: encoding: [0xc5,0xdd,0x59,0x30]
          vmulpd  (%rax), %ymm4, %ymm6

// CHECK: vsqrtpd  %ymm11, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7d,0x51,0xe3]
          vsqrtpd  %ymm11, %ymm12

// CHECK: vsqrtpd  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7d,0x51,0x20]
          vsqrtpd  (%rax), %ymm12

// CHECK: vsqrtps  %ymm11, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7c,0x51,0xe3]
          vsqrtps  %ymm11, %ymm12

// CHECK: vsqrtps  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7c,0x51,0x20]
          vsqrtps  (%rax), %ymm12

// CHECK: vrsqrtps  %ymm11, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7c,0x52,0xe3]
          vrsqrtps  %ymm11, %ymm12

// CHECK: vrsqrtps  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7c,0x52,0x20]
          vrsqrtps  (%rax), %ymm12

// CHECK: vrcpps  %ymm11, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7c,0x53,0xe3]
          vrcpps  %ymm11, %ymm12

// CHECK: vrcpps  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7c,0x53,0x20]
          vrcpps  (%rax), %ymm12

// CHECK: vandps  %ymm12, %ymm14, %ymm11
// CHECK: encoding: [0xc4,0x41,0x0c,0x54,0xdc]
          vandps  %ymm12, %ymm14, %ymm11

// CHECK: vandpd  %ymm12, %ymm14, %ymm11
// CHECK: encoding: [0xc4,0x41,0x0d,0x54,0xdc]
          vandpd  %ymm12, %ymm14, %ymm11

// CHECK: vandps  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1c,0x54,0x54,0xcb,0xfc]
          vandps  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vandpd  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1d,0x54,0x54,0xcb,0xfc]
          vandpd  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vorps  %ymm12, %ymm14, %ymm11
// CHECK: encoding: [0xc4,0x41,0x0c,0x56,0xdc]
          vorps  %ymm12, %ymm14, %ymm11

// CHECK: vorpd  %ymm12, %ymm14, %ymm11
// CHECK: encoding: [0xc4,0x41,0x0d,0x56,0xdc]
          vorpd  %ymm12, %ymm14, %ymm11

// CHECK: vorps  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1c,0x56,0x54,0xcb,0xfc]
          vorps  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vorpd  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1d,0x56,0x54,0xcb,0xfc]
          vorpd  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vxorps  %ymm12, %ymm14, %ymm11
// CHECK: encoding: [0xc4,0x41,0x0c,0x57,0xdc]
          vxorps  %ymm12, %ymm14, %ymm11

// CHECK: vxorpd  %ymm12, %ymm14, %ymm11
// CHECK: encoding: [0xc4,0x41,0x0d,0x57,0xdc]
          vxorpd  %ymm12, %ymm14, %ymm11

// CHECK: vxorps  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1c,0x57,0x54,0xcb,0xfc]
          vxorps  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vxorpd  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1d,0x57,0x54,0xcb,0xfc]
          vxorpd  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vandnps  %ymm12, %ymm14, %ymm11
// CHECK: encoding: [0xc4,0x41,0x0c,0x55,0xdc]
          vandnps  %ymm12, %ymm14, %ymm11

// CHECK: vandnpd  %ymm12, %ymm14, %ymm11
// CHECK: encoding: [0xc4,0x41,0x0d,0x55,0xdc]
          vandnpd  %ymm12, %ymm14, %ymm11

// CHECK: vandnps  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1c,0x55,0x54,0xcb,0xfc]
          vandnps  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vandnpd  -4(%rbx,%rcx,8), %ymm12, %ymm10
// CHECK: encoding: [0xc5,0x1d,0x55,0x54,0xcb,0xfc]
          vandnpd  -4(%rbx,%rcx,8), %ymm12, %ymm10

// CHECK: vcvtps2pd  %xmm13, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7c,0x5a,0xe5]
          vcvtps2pd  %xmm13, %ymm12

// CHECK: vcvtps2pd  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7c,0x5a,0x20]
          vcvtps2pd  (%rax), %ymm12

// CHECK: vcvtdq2pd  %xmm13, %ymm12
// CHECK: encoding: [0xc4,0x41,0x7e,0xe6,0xe5]
          vcvtdq2pd  %xmm13, %ymm12

// CHECK: vcvtdq2pd  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7e,0xe6,0x20]
          vcvtdq2pd  (%rax), %ymm12

// CHECK: vcvtdq2ps  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x41,0x7c,0x5b,0xd4]
          vcvtdq2ps  %ymm12, %ymm10

// CHECK: vcvtdq2ps  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7c,0x5b,0x20]
          vcvtdq2ps  (%rax), %ymm12

// CHECK: vcvtps2dq  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x41,0x7d,0x5b,0xd4]
          vcvtps2dq  %ymm12, %ymm10

// CHECK: vcvtps2dq  (%rax), %ymm10
// CHECK: encoding: [0xc5,0x7d,0x5b,0x10]
          vcvtps2dq  (%rax), %ymm10

// CHECK: vcvttps2dq  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x41,0x7e,0x5b,0xd4]
          vcvttps2dq  %ymm12, %ymm10

// CHECK: vcvttps2dq  (%rax), %ymm10
// CHECK: encoding: [0xc5,0x7e,0x5b,0x10]
          vcvttps2dq  (%rax), %ymm10

// CHECK: vcvttpd2dq  %xmm11, %xmm10
// CHECK: encoding: [0xc4,0x41,0x79,0xe6,0xd3]
          vcvttpd2dq  %xmm11, %xmm10

// CHECK: vcvttpd2dq %ymm12, %xmm10
// CHECK: encoding: [0xc4,0x41,0x7d,0xe6,0xd4]
          vcvttpd2dq  %ymm12, %xmm10

// CHECK: vcvttpd2dq   %xmm11, %xmm10
// CHECK: encoding: [0xc4,0x41,0x79,0xe6,0xd3]
          vcvttpd2dqx  %xmm11, %xmm10

// CHECK: vcvttpd2dqx  (%rax), %xmm11
// CHECK: encoding: [0xc5,0x79,0xe6,0x18]
          vcvttpd2dqx  (%rax), %xmm11

// CHECK: vcvttpd2dq  %ymm12, %xmm11
// CHECK: encoding: [0xc4,0x41,0x7d,0xe6,0xdc]
          vcvttpd2dqy  %ymm12, %xmm11

// CHECK: vcvttpd2dqy  (%rax), %xmm11
// CHECK: encoding: [0xc5,0x7d,0xe6,0x18]
          vcvttpd2dqy  (%rax), %xmm11

// CHECK: vcvtpd2ps %ymm12, %xmm10
// CHECK: encoding: [0xc4,0x41,0x7d,0x5a,0xd4]
          vcvtpd2ps  %ymm12, %xmm10

// CHECK: vcvtpd2ps   %xmm11, %xmm10
// CHECK: encoding: [0xc4,0x41,0x79,0x5a,0xd3]
          vcvtpd2psx  %xmm11, %xmm10

// CHECK: vcvtpd2psx  (%rax), %xmm11
// CHECK: encoding: [0xc5,0x79,0x5a,0x18]
          vcvtpd2psx  (%rax), %xmm11

// CHECK: vcvtpd2ps  %ymm12, %xmm11
// CHECK: encoding: [0xc4,0x41,0x7d,0x5a,0xdc]
          vcvtpd2psy  %ymm12, %xmm11

// CHECK: vcvtpd2psy  (%rax), %xmm11
// CHECK: encoding: [0xc5,0x7d,0x5a,0x18]
          vcvtpd2psy  (%rax), %xmm11

// CHECK: vcvtpd2dq %ymm12, %xmm10
// CHECK: encoding: [0xc4,0x41,0x7f,0xe6,0xd4]
          vcvtpd2dq  %ymm12, %xmm10

// CHECK: vcvtpd2dq  %ymm12, %xmm11
// CHECK: encoding: [0xc4,0x41,0x7f,0xe6,0xdc]
          vcvtpd2dqy  %ymm12, %xmm11

// CHECK: vcvtpd2dqy  (%rax), %xmm11
// CHECK: encoding: [0xc5,0x7f,0xe6,0x18]
          vcvtpd2dqy  (%rax), %xmm11

// CHECK: vcvtpd2dq   %xmm11, %xmm10
// CHECK: encoding: [0xc4,0x41,0x7b,0xe6,0xd3]
          vcvtpd2dqx  %xmm11, %xmm10

// CHECK: vcvtpd2dqx  (%rax), %xmm11
// CHECK: encoding: [0xc5,0x7b,0xe6,0x18]
          vcvtpd2dqx  (%rax), %xmm11

// CHECK: vcmpeqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x00]
          vcmpeqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpleps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x02]
          vcmpleps %ymm11, %ymm12, %ymm13

// CHECK: vcmpltps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x01]
          vcmpltps %ymm11, %ymm12, %ymm13

// CHECK: vcmpneqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x04]
          vcmpneqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpnleps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x06]
          vcmpnleps %ymm11, %ymm12, %ymm13

// CHECK: vcmpnltps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x05]
          vcmpnltps %ymm11, %ymm12, %ymm13

// CHECK: vcmpordps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x07]
          vcmpordps %ymm11, %ymm12, %ymm13

// CHECK: vcmpunordps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x03]
          vcmpunordps %ymm11, %ymm12, %ymm13

// CHECK: vcmpeqps  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1c,0xc2,0x6c,0xcb,0xfc,0x00]
          vcmpeqps -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpleps  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1c,0xc2,0x6c,0xcb,0xfc,0x02]
          vcmpleps -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpltps  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1c,0xc2,0x6c,0xcb,0xfc,0x01]
          vcmpltps -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpneqps  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1c,0xc2,0x6c,0xcb,0xfc,0x04]
          vcmpneqps -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpnleps  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1c,0xc2,0x6c,0xcb,0xfc,0x06]
          vcmpnleps -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpnltps  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1c,0xc2,0x6c,0xcb,0xfc,0x05]
          vcmpnltps -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpordps  -4(%rbx,%rcx,8), %ymm6, %ymm12
// CHECK: encoding: [0xc5,0x4c,0xc2,0x64,0xcb,0xfc,0x07]
          vcmpordps -4(%rbx,%rcx,8), %ymm6, %ymm12

// CHECK: vcmpunordps  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1c,0xc2,0x6c,0xcb,0xfc,0x03]
          vcmpunordps -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpeqpd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0xc2,0xeb,0x00]
          vcmpeqpd %ymm11, %ymm12, %ymm13

// CHECK: vcmplepd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0xc2,0xeb,0x02]
          vcmplepd %ymm11, %ymm12, %ymm13

// CHECK: vcmpltpd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0xc2,0xeb,0x01]
          vcmpltpd %ymm11, %ymm12, %ymm13

// CHECK: vcmpneqpd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0xc2,0xeb,0x04]
          vcmpneqpd %ymm11, %ymm12, %ymm13

// CHECK: vcmpnlepd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0xc2,0xeb,0x06]
          vcmpnlepd %ymm11, %ymm12, %ymm13

// CHECK: vcmpnltpd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0xc2,0xeb,0x05]
          vcmpnltpd %ymm11, %ymm12, %ymm13

// CHECK: vcmpordpd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0xc2,0xeb,0x07]
          vcmpordpd %ymm11, %ymm12, %ymm13

// CHECK: vcmpunordpd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0xc2,0xeb,0x03]
          vcmpunordpd %ymm11, %ymm12, %ymm13

// CHECK: vcmpeqpd  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1d,0xc2,0x6c,0xcb,0xfc,0x00]
          vcmpeqpd -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmplepd  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1d,0xc2,0x6c,0xcb,0xfc,0x02]
          vcmplepd -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpltpd  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1d,0xc2,0x6c,0xcb,0xfc,0x01]
          vcmpltpd -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpneqpd  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1d,0xc2,0x6c,0xcb,0xfc,0x04]
          vcmpneqpd -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpnlepd  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1d,0xc2,0x6c,0xcb,0xfc,0x06]
          vcmpnlepd -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpnltpd  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1d,0xc2,0x6c,0xcb,0xfc,0x05]
          vcmpnltpd -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpordpd  -4(%rbx,%rcx,8), %ymm6, %ymm12
// CHECK: encoding: [0xc5,0x4d,0xc2,0x64,0xcb,0xfc,0x07]
          vcmpordpd -4(%rbx,%rcx,8), %ymm6, %ymm12

// CHECK: vcmpunordpd  -4(%rbx,%rcx,8), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1d,0xc2,0x6c,0xcb,0xfc,0x03]
          vcmpunordpd -4(%rbx,%rcx,8), %ymm12, %ymm13

// CHECK: vcmpeq_uqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x08]
          vcmpeq_uqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpngeps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x09]
          vcmpngeps %ymm11, %ymm12, %ymm13

// CHECK: vcmpngtps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x0a]
          vcmpngtps %ymm11, %ymm12, %ymm13

// CHECK: vcmpfalseps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x0b]
          vcmpfalseps %ymm11, %ymm12, %ymm13

// CHECK: vcmpneq_oqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x0c]
          vcmpneq_oqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpgeps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x0d]
          vcmpgeps %ymm11, %ymm12, %ymm13

// CHECK: vcmpgtps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x0e]
          vcmpgtps %ymm11, %ymm12, %ymm13

// CHECK: vcmptrueps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x0f]
          vcmptrueps %ymm11, %ymm12, %ymm13

// CHECK: vcmpeq_osps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x10]
          vcmpeq_osps %ymm11, %ymm12, %ymm13

// CHECK: vcmplt_oqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x11]
          vcmplt_oqps %ymm11, %ymm12, %ymm13

// CHECK: vcmple_oqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x12]
          vcmple_oqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpunord_sps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x13]
          vcmpunord_sps %ymm11, %ymm12, %ymm13

// CHECK: vcmpneq_usps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x14]
          vcmpneq_usps %ymm11, %ymm12, %ymm13

// CHECK: vcmpnlt_uqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x15]
          vcmpnlt_uqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpnle_uqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x16]
          vcmpnle_uqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpord_sps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x17]
          vcmpord_sps %ymm11, %ymm12, %ymm13

// CHECK: vcmpeq_usps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x18]
          vcmpeq_usps %ymm11, %ymm12, %ymm13

// CHECK: vcmpnge_uqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x19]
          vcmpnge_uqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpngt_uqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x1a]
          vcmpngt_uqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpfalse_osps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x1b]
          vcmpfalse_osps %ymm11, %ymm12, %ymm13

// CHECK: vcmpneq_osps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x1c]
          vcmpneq_osps %ymm11, %ymm12, %ymm13

// CHECK: vcmpge_oqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x1d]
          vcmpge_oqps %ymm11, %ymm12, %ymm13

// CHECK: vcmpgt_oqps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x1e]
          vcmpgt_oqps %ymm11, %ymm12, %ymm13

// CHECK: vcmptrue_usps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1c,0xc2,0xeb,0x1f]
          vcmptrue_usps %ymm11, %ymm12, %ymm13

// CHECK: vaddsubps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1f,0xd0,0xeb]
          vaddsubps  %ymm11, %ymm12, %ymm13

// CHECK: vaddsubps  (%rax), %ymm11, %ymm12
// CHECK: encoding: [0xc5,0x27,0xd0,0x20]
          vaddsubps  (%rax), %ymm11, %ymm12

// CHECK: vaddsubpd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0xd0,0xeb]
          vaddsubpd  %ymm11, %ymm12, %ymm13

// CHECK: vaddsubpd  (%rax), %ymm11, %ymm12
// CHECK: encoding: [0xc5,0x25,0xd0,0x20]
          vaddsubpd  (%rax), %ymm11, %ymm12

// CHECK: vhaddps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1f,0x7c,0xeb]
          vhaddps  %ymm11, %ymm12, %ymm13

// CHECK: vhaddps  (%rax), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1f,0x7c,0x28]
          vhaddps  (%rax), %ymm12, %ymm13

// CHECK: vhaddpd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0x7c,0xeb]
          vhaddpd  %ymm11, %ymm12, %ymm13

// CHECK: vhaddpd  (%rax), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1d,0x7c,0x28]
          vhaddpd  (%rax), %ymm12, %ymm13

// CHECK: vhsubps  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1f,0x7d,0xeb]
          vhsubps  %ymm11, %ymm12, %ymm13

// CHECK: vhsubps  (%rax), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1f,0x7d,0x28]
          vhsubps  (%rax), %ymm12, %ymm13

// CHECK: vhsubpd  %ymm11, %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x41,0x1d,0x7d,0xeb]
          vhsubpd  %ymm11, %ymm12, %ymm13

// CHECK: vhsubpd  (%rax), %ymm12, %ymm13
// CHECK: encoding: [0xc5,0x1d,0x7d,0x28]
          vhsubpd  (%rax), %ymm12, %ymm13

// CHECK: vblendps  $3, %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x43,0x2d,0x0c,0xdc,0x03]
          vblendps  $3, %ymm12, %ymm10, %ymm11

// CHECK: vblendps  $3, (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x63,0x2d,0x0c,0x18,0x03]
          vblendps  $3, (%rax), %ymm10, %ymm11

// CHECK: vblendpd  $3, %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x43,0x2d,0x0d,0xdc,0x03]
          vblendpd  $3, %ymm12, %ymm10, %ymm11

// CHECK: vblendpd  $3, (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x63,0x2d,0x0d,0x18,0x03]
          vblendpd  $3, (%rax), %ymm10, %ymm11

// CHECK: vdpps  $3, %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x43,0x2d,0x40,0xdc,0x03]
          vdpps  $3, %ymm12, %ymm10, %ymm11

// CHECK: vdpps  $3, (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x63,0x2d,0x40,0x18,0x03]
          vdpps  $3, (%rax), %ymm10, %ymm11

// CHECK: vbroadcasti128  (%rax), %ymm12
// CHECK: encoding: [0xc4,0x62,0x7d,0x5a,0x20]
          vbroadcasti128  (%rax), %ymm12

// CHECK: vbroadcastf128  (%rax), %ymm12
// CHECK: encoding: [0xc4,0x62,0x7d,0x1a,0x20]
          vbroadcastf128  (%rax), %ymm12

// CHECK: vbroadcastsd  (%rax), %ymm12
// CHECK: encoding: [0xc4,0x62,0x7d,0x19,0x20]
          vbroadcastsd  (%rax), %ymm12

// CHECK: vbroadcastss  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x18,0x20]
          vbroadcastss  (%rax), %xmm12

// CHECK: vbroadcastss  (%rax), %ymm12
// CHECK: encoding: [0xc4,0x62,0x7d,0x18,0x20]
          vbroadcastss  (%rax), %ymm12

// CHECK: vinsertf128  $7, %xmm12, %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x43,0x1d,0x18,0xd4,0x07]
          vinsertf128  $7, %xmm12, %ymm12, %ymm10

// CHECK: vinsertf128  $7, (%rax), %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x63,0x1d,0x18,0x10,0x07]
          vinsertf128  $7, (%rax), %ymm12, %ymm10

// CHECK: vextractf128  $7, %ymm12, %xmm12
// CHECK: encoding: [0xc4,0x43,0x7d,0x19,0xe4,0x07]
          vextractf128  $7, %ymm12, %xmm12

// CHECK: vextractf128  $7, %ymm12, (%rax)
// CHECK: encoding: [0xc4,0x63,0x7d,0x19,0x20,0x07]
          vextractf128  $7, %ymm12, (%rax)

// CHECK: vmaskmovpd  %xmm12, %xmm10, (%rax)
// CHECK: encoding: [0xc4,0x62,0x29,0x2f,0x20]
          vmaskmovpd  %xmm12, %xmm10, (%rax)

// CHECK: vmaskmovpd  %ymm12, %ymm10, (%rax)
// CHECK: encoding: [0xc4,0x62,0x2d,0x2f,0x20]
          vmaskmovpd  %ymm12, %ymm10, (%rax)

// CHECK: vmaskmovpd  (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x62,0x19,0x2d,0x10]
          vmaskmovpd  (%rax), %xmm12, %xmm10

// CHECK: vmaskmovpd  (%rax), %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x62,0x1d,0x2d,0x10]
          vmaskmovpd  (%rax), %ymm12, %ymm10

// CHECK: vmaskmovps  %xmm12, %xmm10, (%rax)
// CHECK: encoding: [0xc4,0x62,0x29,0x2e,0x20]
          vmaskmovps  %xmm12, %xmm10, (%rax)

// CHECK: vmaskmovps  %ymm12, %ymm10, (%rax)
// CHECK: encoding: [0xc4,0x62,0x2d,0x2e,0x20]
          vmaskmovps  %ymm12, %ymm10, (%rax)

// CHECK: vmaskmovps  (%rax), %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x62,0x19,0x2c,0x10]
          vmaskmovps  (%rax), %xmm12, %xmm10

// CHECK: vmaskmovps  (%rax), %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x62,0x1d,0x2c,0x10]
          vmaskmovps  (%rax), %ymm12, %ymm10

// CHECK: vpermilps  $7, %xmm11, %xmm10
// CHECK: encoding: [0xc4,0x43,0x79,0x04,0xd3,0x07]
          vpermilps  $7, %xmm11, %xmm10

// CHECK: vpermilps  $7, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x43,0x7d,0x04,0xda,0x07]
          vpermilps  $7, %ymm10, %ymm11

// CHECK: vpermilps  $7, (%rax), %xmm10
// CHECK: encoding: [0xc4,0x63,0x79,0x04,0x10,0x07]
          vpermilps  $7, (%rax), %xmm10

// CHECK: vpermilps  $7, (%rax), %ymm10
// CHECK: encoding: [0xc4,0x63,0x7d,0x04,0x10,0x07]
          vpermilps  $7, (%rax), %ymm10

// CHECK: vpermilps  %xmm11, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x0c,0xdb]
          vpermilps  %xmm11, %xmm10, %xmm11

// CHECK: vpermilps  %ymm11, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0x0c,0xdb]
          vpermilps  %ymm11, %ymm10, %ymm11

// CHECK: vpermilps  (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x62,0x29,0x0c,0x28]
          vpermilps  (%rax), %xmm10, %xmm13

// CHECK: vpermilps  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0x0c,0x18]
          vpermilps  (%rax), %ymm10, %ymm11

// CHECK: vpermilpd  $7, %xmm11, %xmm10
// CHECK: encoding: [0xc4,0x43,0x79,0x05,0xd3,0x07]
          vpermilpd  $7, %xmm11, %xmm10

// CHECK: vpermilpd  $7, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x43,0x7d,0x05,0xda,0x07]
          vpermilpd  $7, %ymm10, %ymm11

// CHECK: vpermilpd  $7, (%rax), %xmm10
// CHECK: encoding: [0xc4,0x63,0x79,0x05,0x10,0x07]
          vpermilpd  $7, (%rax), %xmm10

// CHECK: vpermilpd  $7, (%rax), %ymm10
// CHECK: encoding: [0xc4,0x63,0x7d,0x05,0x10,0x07]
          vpermilpd  $7, (%rax), %ymm10

// CHECK: vpermilpd  %xmm11, %xmm10, %xmm11
// CHECK: encoding: [0xc4,0x42,0x29,0x0d,0xdb]
          vpermilpd  %xmm11, %xmm10, %xmm11

// CHECK: vpermilpd  %ymm11, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x42,0x2d,0x0d,0xdb]
          vpermilpd  %ymm11, %ymm10, %ymm11

// CHECK: vpermilpd  (%rax), %xmm10, %xmm13
// CHECK: encoding: [0xc4,0x62,0x29,0x0d,0x28]
          vpermilpd  (%rax), %xmm10, %xmm13

// CHECK: vpermilpd  (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x62,0x2d,0x0d,0x18]
          vpermilpd  (%rax), %ymm10, %ymm11

// CHECK: vperm2f128  $7, %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x43,0x2d,0x06,0xdc,0x07]
          vperm2f128  $7, %ymm12, %ymm10, %ymm11

// CHECK: vperm2f128  $7, (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x63,0x2d,0x06,0x18,0x07]
          vperm2f128  $7, (%rax), %ymm10, %ymm11

// CHECK: vcvtsd2si  %xmm8, %r8d
// CHECK: encoding: [0xc4,0x41,0x7b,0x2d,0xc0]
          vcvtsd2si  %xmm8, %r8d

// CHECK: vcvtsd2si  (%rcx), %ecx
// CHECK: encoding: [0xc5,0xfb,0x2d,0x09]
          vcvtsd2si  (%rcx), %ecx

// CHECK: vcvtss2si  %xmm4, %rcx
// CHECK: encoding: [0xc4,0xe1,0xfa,0x2d,0xcc]
          vcvtss2si  %xmm4, %rcx

// CHECK: vcvtss2si  (%rcx), %r8
// CHECK: encoding: [0xc4,0x61,0xfa,0x2d,0x01]
          vcvtss2si  (%rcx), %r8

// CHECK: vcvtsi2sd  %r8d, %xmm8, %xmm15
// CHECK: encoding: [0xc4,0x41,0x3b,0x2a,0xf8]
          vcvtsi2sdl  %r8d, %xmm8, %xmm15

// CHECK: vcvtsi2sdl  (%rbp), %xmm8, %xmm15
// CHECK: encoding: [0xc5,0x3b,0x2a,0x7d,0x00]
          vcvtsi2sdl  (%rbp), %xmm8, %xmm15

// CHECK: vcvtsi2sd  %rcx, %xmm4, %xmm6
// CHECK: encoding: [0xc4,0xe1,0xdb,0x2a,0xf1]
          vcvtsi2sdq  %rcx, %xmm4, %xmm6

// CHECK: vcvtsi2sdq  (%rcx), %xmm4, %xmm6
// CHECK: encoding: [0xc4,0xe1,0xdb,0x2a,0x31]
          vcvtsi2sdq  (%rcx), %xmm4, %xmm6

// CHECK: vcvtsi2ss  %rcx, %xmm4, %xmm6
// CHECK: encoding: [0xc4,0xe1,0xda,0x2a,0xf1]
          vcvtsi2ssq  %rcx, %xmm4, %xmm6

// CHECK: vcvtsi2ssq  (%rcx), %xmm4, %xmm6
// CHECK: encoding: [0xc4,0xe1,0xda,0x2a,0x31]
          vcvtsi2ssq  (%rcx), %xmm4, %xmm6

// CHECK: vcvttsd2si  %xmm4, %rcx
// CHECK: encoding: [0xc4,0xe1,0xfb,0x2c,0xcc]
          vcvttsd2si  %xmm4, %rcx

// CHECK: vcvttsd2si  (%rcx), %rcx
// CHECK: encoding: [0xc4,0xe1,0xfb,0x2c,0x09]
          vcvttsd2si  (%rcx), %rcx

// CHECK: vcvttss2si  %xmm4, %rcx
// CHECK: encoding: [0xc4,0xe1,0xfa,0x2c,0xcc]
          vcvttss2si  %xmm4, %rcx

// CHECK: vcvttss2si  (%rcx), %rcx
// CHECK: encoding: [0xc4,0xe1,0xfa,0x2c,0x09]
          vcvttss2si  (%rcx), %rcx

// CHECK: vlddqu  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7f,0xf0,0x20]
          vlddqu  (%rax), %ymm12

// CHECK: vmovddup  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x41,0x7f,0x12,0xd4]
          vmovddup  %ymm12, %ymm10

// CHECK: vmovddup  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7f,0x12,0x20]
          vmovddup  (%rax), %ymm12

// CHECK: vmovdqa  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x41,0x7d,0x6f,0xd4]
          vmovdqa  %ymm12, %ymm10

// CHECK: vmovdqa  %ymm12, (%rax)
// CHECK: encoding: [0xc5,0x7d,0x7f,0x20]
          vmovdqa  %ymm12, (%rax)

// CHECK: vmovdqa  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7d,0x6f,0x20]
          vmovdqa  (%rax), %ymm12

// CHECK: vmovdqu  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x41,0x7e,0x6f,0xd4]
          vmovdqu  %ymm12, %ymm10

// CHECK: vmovdqu  %ymm12, (%rax)
// CHECK: encoding: [0xc5,0x7e,0x7f,0x20]
          vmovdqu  %ymm12, (%rax)

// CHECK: vmovdqu  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7e,0x6f,0x20]
          vmovdqu  (%rax), %ymm12

// CHECK: vmovshdup  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x41,0x7e,0x16,0xd4]
          vmovshdup  %ymm12, %ymm10

// CHECK: vmovshdup  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7e,0x16,0x20]
          vmovshdup  (%rax), %ymm12

// CHECK: vmovsldup  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x41,0x7e,0x12,0xd4]
          vmovsldup  %ymm12, %ymm10

// CHECK: vmovsldup  (%rax), %ymm12
// CHECK: encoding: [0xc5,0x7e,0x12,0x20]
          vmovsldup  (%rax), %ymm12

// CHECK: vptest  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x42,0x7d,0x17,0xd4]
          vptest  %ymm12, %ymm10

// CHECK: vptest  (%rax), %ymm12
// CHECK: encoding: [0xc4,0x62,0x7d,0x17,0x20]
          vptest  (%rax), %ymm12

// CHECK: vroundpd  $7, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x43,0x7d,0x09,0xda,0x07]
          vroundpd  $7, %ymm10, %ymm11

// CHECK: vroundpd  $7, (%rax), %ymm10
// CHECK: encoding: [0xc4,0x63,0x7d,0x09,0x10,0x07]
          vroundpd  $7, (%rax), %ymm10

// CHECK: vroundps  $7, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x43,0x7d,0x08,0xda,0x07]
          vroundps  $7, %ymm10, %ymm11

// CHECK: vroundps  $7, (%rax), %ymm10
// CHECK: encoding: [0xc4,0x63,0x7d,0x08,0x10,0x07]
          vroundps  $7, (%rax), %ymm10

// CHECK: vshufpd  $7, %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x41,0x2d,0xc6,0xdc,0x07]
          vshufpd  $7, %ymm12, %ymm10, %ymm11

// CHECK: vshufpd  $7, (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc5,0x2d,0xc6,0x18,0x07]
          vshufpd  $7, (%rax), %ymm10, %ymm11

// CHECK: vshufps  $7, %ymm12, %ymm10, %ymm11
// CHECK: encoding: [0xc4,0x41,0x2c,0xc6,0xdc,0x07]
          vshufps  $7, %ymm12, %ymm10, %ymm11

// CHECK: vshufps  $7, (%rax), %ymm10, %ymm11
// CHECK: encoding: [0xc5,0x2c,0xc6,0x18,0x07]
          vshufps  $7, (%rax), %ymm10, %ymm11

// CHECK: vtestpd  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x0f,0xd4]
          vtestpd  %xmm12, %xmm10

// CHECK: vtestpd  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x42,0x7d,0x0f,0xd4]
          vtestpd  %ymm12, %ymm10

// CHECK: vtestpd  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x0f,0x20]
          vtestpd  (%rax), %xmm12

// CHECK: vtestpd  (%rax), %ymm12
// CHECK: encoding: [0xc4,0x62,0x7d,0x0f,0x20]
          vtestpd  (%rax), %ymm12

// CHECK: vtestps  %xmm12, %xmm10
// CHECK: encoding: [0xc4,0x42,0x79,0x0e,0xd4]
          vtestps  %xmm12, %xmm10

// CHECK: vtestps  %ymm12, %ymm10
// CHECK: encoding: [0xc4,0x42,0x7d,0x0e,0xd4]
          vtestps  %ymm12, %ymm10

// CHECK: vtestps  (%rax), %xmm12
// CHECK: encoding: [0xc4,0x62,0x79,0x0e,0x20]
          vtestps  (%rax), %xmm12

// CHECK: vtestps  (%rax), %ymm12
// CHECK: encoding: [0xc4,0x62,0x7d,0x0e,0x20]
          vtestps  (%rax), %ymm12

// CHECK: vextractps   $10, %xmm8, %r8
// CHECK: encoding: [0xc4,0x43,0x79,0x17,0xc0,0x0a]
          vextractps   $10, %xmm8, %r8

// CHECK: vextractps   $7, %xmm4, %ecx
// CHECK: encoding: [0xc4,0xe3,0x79,0x17,0xe1,0x07]
          vextractps   $7, %xmm4, %rcx

// CHECK: vmovq  %xmm4, %rcx
// CHECK: encoding: [0xc4,0xe1,0xf9,0x7e,0xe1]
          vmovd  %xmm4, %rcx

// CHECK: vmovmskpd  %xmm4, %ecx
// CHECK: encoding: [0xc5,0xf9,0x50,0xcc]
          vmovmskpd  %xmm4, %rcx

// CHECK: vmovmskpd  %ymm4, %ecx
// CHECK: encoding: [0xc5,0xfd,0x50,0xcc]
          vmovmskpd  %ymm4, %rcx

// CHECK: vmovmskps  %xmm4, %ecx
// CHECK: encoding: [0xc5,0xf8,0x50,0xcc]
          vmovmskps  %xmm4, %rcx

// CHECK: vmovmskps  %ymm4, %ecx
// CHECK: encoding: [0xc5,0xfc,0x50,0xcc]
          vmovmskps  %ymm4, %rcx

// CHECK: vpextrb  $7, %xmm4, %ecx
// CHECK: encoding: [0xc4,0xe3,0x79,0x14,0xe1,0x07]
          vpextrb  $7, %xmm4, %rcx

// CHECK: vpinsrw  $7, %r8d, %xmm15, %xmm8
// CHECK: encoding: [0xc4,0x41,0x01,0xc4,0xc0,0x07]
          vpinsrw  $7, %r8, %xmm15, %xmm8

// CHECK: vpinsrw  $7, %ecx, %xmm4, %xmm6
// CHECK: encoding: [0xc5,0xd9,0xc4,0xf1,0x07]
          vpinsrw  $7, %rcx, %xmm4, %xmm6

// CHECK: vpmovmskb  %xmm4, %ecx
// CHECK: encoding: [0xc5,0xf9,0xd7,0xcc]
          vpmovmskb  %xmm4, %rcx

// CHECK: vblendvpd  %ymm11, 57005(%rax,%riz), %ymm12, %ymm13
// CHECK: encoding: [0xc4,0x63,0x1d,0x4b,0xac,0x20,0xad,0xde,0x00,0x00,0xb0]
          vblendvpd  %ymm11, 0xdead(%rax,%riz), %ymm12, %ymm13

// CHECK: vmovaps	%xmm3, (%r14,%r11)
// CHECK: encoding: [0xc4,0x81,0x78,0x29,0x1c,0x1e]
          vmovaps	%xmm3, (%r14,%r11)

// CHECK: vmovaps	(%r14,%r11), %xmm3
// CHECK: encoding: [0xc4,0x81,0x78,0x28,0x1c,0x1e]
          vmovaps	(%r14,%r11), %xmm3

// CHECK: vmovaps	%xmm3, (%r14,%rbx)
// CHECK: encoding: [0xc4,0xc1,0x78,0x29,0x1c,0x1e]
          vmovaps	%xmm3, (%r14,%rbx)

// CHECK: vmovaps	(%r14,%rbx), %xmm3
// CHECK: encoding: [0xc4,0xc1,0x78,0x28,0x1c,0x1e]
          vmovaps	(%r14,%rbx), %xmm3

// CHECK: vmovaps %xmm3, (%rax,%r11)
// CHECK: encoding: [0xc4,0xa1,0x78,0x29,0x1c,0x18]
          vmovaps %xmm3, (%rax,%r11)

// CHECK: vpshufb _foo(%rip), %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xe2,0x79,0x00,0x05,A,A,A,A]
// CHECK: kind: reloc_riprel_4byte
_foo:
  nop
  vpshufb _foo(%rip), %xmm0, %xmm0

// CHECK: vblendvps %ymm1, _foo2(%rip), %ymm0, %ymm0
// CHECK: encoding: [0xc4,0xe3,0x7d,0x4a,0x05,A,A,A,A,0x10]
// CHECK: fixup A - offset: 5, value: _foo2-5
_foo2:
  nop
  vblendvps %ymm1, _foo2(%rip), %ymm0, %ymm0

// CHECK: vgatherdpd %xmm0, (%rdi,%xmm1,2), %xmm2
// CHECK: encoding: [0xc4,0xe2,0xf9,0x92,0x14,0x4f]
          vgatherdpd %xmm0, (%rdi,%xmm1,2), %xmm2

// CHECK: vgatherqpd %xmm0, (%rdi,%xmm1,2), %xmm2
// CHECK: encoding: [0xc4,0xe2,0xf9,0x93,0x14,0x4f]
          vgatherqpd %xmm0, (%rdi,%xmm1,2), %xmm2

// CHECK: vgatherdpd %ymm0, (%rdi,%xmm1,2), %ymm2
// CHECK: encoding: [0xc4,0xe2,0xfd,0x92,0x14,0x4f]
          vgatherdpd %ymm0, (%rdi,%xmm1,2), %ymm2

// CHECK: vgatherqpd %ymm0, (%rdi,%ymm1,2), %ymm2
// CHECK: encoding: [0xc4,0xe2,0xfd,0x93,0x14,0x4f]
          vgatherqpd %ymm0, (%rdi,%ymm1,2), %ymm2

// CHECK: vgatherdps %xmm8, (%r15,%xmm9,2), %xmm10
// CHECK: encoding: [0xc4,0x02,0x39,0x92,0x14,0x4f]
          vgatherdps %xmm8, (%r15,%xmm9,2), %xmm10

// CHECK: vgatherqps %xmm8, (%r15,%xmm9,2), %xmm10
// CHECK: encoding: [0xc4,0x02,0x39,0x93,0x14,0x4f]
          vgatherqps %xmm8, (%r15,%xmm9,2), %xmm10

// CHECK: vgatherdps %ymm8, (%r15,%ymm9,2), %ymm10
// CHECK: encoding: [0xc4,0x02,0x3d,0x92,0x14,0x4f]
          vgatherdps %ymm8, (%r15,%ymm9,2), %ymm10

// CHECK: vgatherqps %xmm8, (%r15,%ymm9,2), %xmm10
// CHECK: encoding: [0xc4,0x02,0x3d,0x93,0x14,0x4f]
          vgatherqps %xmm8, (%r15,%ymm9,2), %xmm10

// CHECK: vpgatherdq %xmm0, (%rdi,%xmm1,2), %xmm2
// CHECK: encoding: [0xc4,0xe2,0xf9,0x90,0x14,0x4f]
          vpgatherdq %xmm0, (%rdi,%xmm1,2), %xmm2

// CHECK: vpgatherqq %xmm0, (%rdi,%xmm1,2), %xmm2
// CHECK: encoding: [0xc4,0xe2,0xf9,0x91,0x14,0x4f]
          vpgatherqq %xmm0, (%rdi,%xmm1,2), %xmm2

// CHECK: vpgatherdq %ymm0, (%rdi,%xmm1,2), %ymm2
// CHECK: encoding: [0xc4,0xe2,0xfd,0x90,0x14,0x4f]
          vpgatherdq %ymm0, (%rdi,%xmm1,2), %ymm2

// CHECK: vpgatherqq %ymm0, (%rdi,%ymm1,2), %ymm2
// CHECK: encoding: [0xc4,0xe2,0xfd,0x91,0x14,0x4f]
          vpgatherqq %ymm0, (%rdi,%ymm1,2), %ymm2

// CHECK: vpgatherdd %xmm8, (%r15,%xmm9,2), %xmm10
// CHECK: encoding: [0xc4,0x02,0x39,0x90,0x14,0x4f]
          vpgatherdd %xmm8, (%r15,%xmm9,2), %xmm10

// CHECK: vpgatherqd %xmm8, (%r15,%xmm9,2), %xmm10
// CHECK: encoding: [0xc4,0x02,0x39,0x91,0x14,0x4f]
          vpgatherqd %xmm8, (%r15,%xmm9,2), %xmm10

// CHECK: vpgatherdd %ymm8, (%r15,%ymm9,2), %ymm10
// CHECK: encoding: [0xc4,0x02,0x3d,0x90,0x14,0x4f]
          vpgatherdd %ymm8, (%r15,%ymm9,2), %ymm10

// CHECK: vpgatherqd %xmm8, (%r15,%ymm9,2), %xmm10
// CHECK: encoding: [0xc4,0x02,0x3d,0x91,0x14,0x4f]
          vpgatherqd %xmm8, (%r15,%ymm9,2), %xmm10

// CHECK: vmovq %xmm0, %xmm8
// CHECK: encoding: [0xc5,0x7a,0x7e,0xc0]
          vmovq %xmm0, %xmm8

// CHECK: vmovq %xmm0, %xmm8
// CHECK: encoding: [0xc4,0x61,0x7a,0x7e,0xc0]
          {vex3} vmovq %xmm0, %xmm8

// CHECK: vmovq %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x79,0xd6,0xc0]
          vmovq %xmm8, %xmm0

// CHECK: vmovq %xmm8, %xmm0
// CHECK: encoding: [0xc4,0xc1,0x7a,0x7e,0xc0]
          {vex3} vmovq %xmm8, %xmm0

// CHECK: vmovdqa %xmm0, %xmm8
// CHECK: encoding: [0xc5,0x79,0x6f,0xc0]
          vmovdqa %xmm0, %xmm8

// CHECK: vmovdqa %xmm0, %xmm8
// CHECK: encoding: [0xc4,0x61,0x79,0x6f,0xc0]
          {vex3} vmovdqa %xmm0, %xmm8

// CHECK: vmovdqa %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x79,0x7f,0xc0]
          vmovdqa %xmm8, %xmm0

// CHECK: vmovdqa %xmm8, %xmm0
// CHECK: encoding: [0xc4,0xc1,0x79,0x6f,0xc0]
          {vex3} vmovdqa %xmm8, %xmm0

// CHECK: vmovdqu %xmm0, %xmm8
// CHECK: encoding: [0xc5,0x7a,0x6f,0xc0]
          vmovdqu %xmm0, %xmm8

// CHECK: vmovdqu %xmm0, %xmm8
// CHECK: encoding: [0xc4,0x61,0x7a,0x6f,0xc0]
          {vex3} vmovdqu %xmm0, %xmm8

// CHECK: vmovdqu %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x7a,0x7f,0xc0]
          vmovdqu %xmm8, %xmm0

// CHECK: vmovdqu %xmm8, %xmm0
// CHECK: encoding: [0xc4,0xc1,0x7a,0x6f,0xc0]
          {vex3} vmovdqu %xmm8, %xmm0

// CHECK: vmovaps %xmm0, %xmm8
// CHECK: encoding: [0xc5,0x78,0x28,0xc0]
          vmovaps %xmm0, %xmm8

// CHECK: vmovaps %xmm0, %xmm8
// CHECK: encoding: [0xc4,0x61,0x78,0x28,0xc0]
          {vex3} vmovaps %xmm0, %xmm8

// CHECK: vmovaps %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x78,0x29,0xc0]
          vmovaps %xmm8, %xmm0

// CHECK: vmovaps %xmm8, %xmm0
// CHECK: encoding: [0xc4,0xc1,0x78,0x28,0xc0]
          {vex3} vmovaps %xmm8, %xmm0

// CHECK: vmovaps %ymm0, %ymm8
// CHECK: encoding: [0xc5,0x7c,0x28,0xc0]
          vmovaps %ymm0, %ymm8

// CHECK: vmovaps %ymm0, %ymm8
// CHECK: encoding: [0xc4,0x61,0x7c,0x28,0xc0]
          {vex3} vmovaps %ymm0, %ymm8

// CHECK: vmovaps %ymm8, %ymm0
// CHECK: encoding: [0xc5,0x7c,0x29,0xc0]
          vmovaps %ymm8, %ymm0

// CHECK: vmovaps %ymm8, %ymm0
// CHECK: encoding: [0xc4,0xc1,0x7c,0x28,0xc0]
          {vex3} vmovaps %ymm8, %ymm0

// CHECK: vmovups %xmm0, %xmm8
// CHECK: encoding: [0xc5,0x78,0x10,0xc0]
          vmovups %xmm0, %xmm8

// CHECK: vmovups %xmm0, %xmm8
// CHECK: encoding: [0xc4,0x61,0x78,0x10,0xc0]
          {vex3} vmovups %xmm0, %xmm8

// CHECK: vmovups %xmm8, %xmm0
// CHECK: encoding: [0xc5,0x78,0x11,0xc0]
          vmovups %xmm8, %xmm0

// CHECK: vmovups %xmm8, %xmm0
// CHECK: encoding: [0xc4,0xc1,0x78,0x10,0xc0]
          {vex3} vmovups %xmm8, %xmm0

// CHECK: vmovups %ymm0, %ymm8
// CHECK: encoding: [0xc5,0x7c,0x10,0xc0]
          vmovups %ymm0, %ymm8

// CHECK: vmovups %ymm0, %ymm8
// CHECK: encoding: [0xc4,0x61,0x7c,0x10,0xc0]
          {vex3} vmovups %ymm0, %ymm8

// CHECK: vmovups %ymm8, %ymm0
// CHECK: encoding: [0xc5,0x7c,0x11,0xc0]
          vmovups %ymm8, %ymm0

// CHECK: vmovups %ymm8, %ymm0
// CHECK: encoding: [0xc4,0xc1,0x7c,0x10,0xc0]
          {vex3} vmovups %ymm8, %ymm0

// CHECK: vmovss %xmm0, %xmm0, %xmm8
// CHECK: encoding: [0xc5,0x7a,0x10,0xc0]
          vmovss %xmm0, %xmm0, %xmm8

// CHECK: vmovss %xmm0, %xmm0, %xmm8
// CHECK: encoding: [0xc4,0x61,0x7a,0x10,0xc0]
          {vex3} vmovss %xmm0, %xmm0, %xmm8

// CHECK: vmovss %xmm0, %xmm8, %xmm0
// CHECK: encoding: [0xc5,0xba,0x10,0xc0]
          vmovss %xmm0, %xmm8, %xmm0

// CHECK: vmovss %xmm0, %xmm8, %xmm0
// CHECK: encoding: [0xc4,0xe1,0x3a,0x10,0xc0]
          {vex3} vmovss %xmm0, %xmm8, %xmm0

// CHECK: vmovss %xmm8, %xmm0, %xmm0
// CHECK: encoding: [0xc5,0x7a,0x11,0xc0]
          vmovss %xmm8, %xmm0, %xmm0

// CHECK: vmovss %xmm8, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xc1,0x7a,0x10,0xc0]
          {vex3} vmovss %xmm8, %xmm0, %xmm0

// CHECK: vmovsd %xmm0, %xmm0, %xmm8
// CHECK: encoding: [0xc5,0x7b,0x10,0xc0]
          vmovsd %xmm0, %xmm0, %xmm8

// CHECK: vmovsd %xmm0, %xmm0, %xmm8
// CHECK: encoding: [0xc4,0x61,0x7b,0x10,0xc0]
          {vex3} vmovsd %xmm0, %xmm0, %xmm8

// CHECK: vmovsd %xmm0, %xmm8, %xmm0
// CHECK: encoding: [0xc5,0xbb,0x10,0xc0]
          vmovsd %xmm0, %xmm8, %xmm0

// CHECK: vmovsd %xmm0, %xmm8, %xmm0
// CHECK: encoding: [0xc4,0xe1,0x3b,0x10,0xc0]
          {vex3} vmovsd %xmm0, %xmm8, %xmm0

// CHECK: vmovsd %xmm8, %xmm0, %xmm0
// CHECK: encoding: [0xc5,0x7b,0x11,0xc0]
          vmovsd %xmm8, %xmm0, %xmm0

// CHECK: vmovsd %xmm8, %xmm0, %xmm0
// CHECK: encoding: [0xc4,0xc1,0x7b,0x10,0xc0]
          {vex3} vmovsd %xmm8, %xmm0, %xmm0

// CHECK: vpmaddwd %xmm3, %xmm2, %xmm1
// CHECK: encoding: [0xc5,0xe9,0xf5,0xcb]
          vpmaddwd %xmm3, %xmm2, %xmm1

// CHECK: vpmaddwd  (%rcx), %xmm2, %xmm1
// CHECK: encoding: [0xc5,0xe9,0xf5,0x09]
          vpmaddwd  (%rcx), %xmm2, %xmm1

// CHECK: vpmaddwd  -4(%rsp), %xmm2, %xmm1
// CHECK: encoding: [0xc5,0xe9,0xf5,0x4c,0x24,0xfc]
          vpmaddwd  -4(%rsp), %xmm2, %xmm1

// CHECK: vpmaddwd  4(%rsp), %xmm2, %xmm1
// CHECK: encoding: [0xc5,0xe9,0xf5,0x4c,0x24,0x04]
          vpmaddwd  4(%rsp), %xmm2, %xmm1

// CHECK: vpmaddwd  268435456(%rcx,%r14,8), %xmm2, %xmm1
// CHECK: encoding: [0xc4,0xa1,0x69,0xf5,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vpmaddwd  268435456(%rcx,%r14,8), %xmm2, %xmm1

// CHECK: vpmaddwd  -536870912(%rcx,%r14,8), %xmm2, %xmm1
// CHECK: encoding: [0xc4,0xa1,0x69,0xf5,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vpmaddwd  -536870912(%rcx,%r14,8), %xmm2, %xmm1

// CHECK: vpmaddwd  -536870910(%rcx,%r14,8), %xmm2, %xmm1
// CHECK: encoding: [0xc4,0xa1,0x69,0xf5,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vpmaddwd  -536870910(%rcx,%r14,8), %xmm2, %xmm1

// CHECK: vpmaddwd %ymm3, %ymm2, %ymm1
// CHECK: encoding: [0xc5,0xed,0xf5,0xcb]
          vpmaddwd %ymm3, %ymm2, %ymm1

// CHECK: vpmaddwd  (%rcx), %ymm2, %ymm1
// CHECK: encoding: [0xc5,0xed,0xf5,0x09]
          vpmaddwd  (%rcx), %ymm2, %ymm1

// CHECK: vpmaddwd  -4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc5,0xed,0xf5,0x4c,0x24,0xfc]
          vpmaddwd  -4(%rsp), %ymm2, %ymm1

// CHECK: vpmaddwd  4(%rsp), %ymm2, %ymm1
// CHECK: encoding: [0xc5,0xed,0xf5,0x4c,0x24,0x04]
          vpmaddwd  4(%rsp), %ymm2, %ymm1

// CHECK: vpmaddwd  268435456(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa1,0x6d,0xf5,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vpmaddwd  268435456(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vpmaddwd  -536870912(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa1,0x6d,0xf5,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vpmaddwd  -536870912(%rcx,%r14,8), %ymm2, %ymm1

// CHECK: vpmaddwd  -536870910(%rcx,%r14,8), %ymm2, %ymm1
// CHECK: encoding: [0xc4,0xa1,0x6d,0xf5,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vpmaddwd  -536870910(%rcx,%r14,8), %ymm2, %ymm1


