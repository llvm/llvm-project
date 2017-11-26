// RUN: llvm-mc -triple x86_64-unknown-unknown -mattr=+gfni --show-encoding < %s | FileCheck %s

// CHECK: gf2p8affineinvqb $7, %xmm2, %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0xcf,0xca,0x07]
          gf2p8affineinvqb $7, %xmm2, %xmm1

// CHECK: gf2p8affineqb $7, %xmm2, %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0xce,0xca,0x07]
          gf2p8affineqb $7, %xmm2, %xmm1

// CHECK: gf2p8affineinvqb  $7, (%rcx), %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0xcf,0x09,0x07]
          gf2p8affineinvqb  $7, (%rcx), %xmm1

// CHECK: gf2p8affineinvqb  $7, -4(%rsp), %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0xcf,0x4c,0x24,0xfc,0x07]
          gf2p8affineinvqb  $7, -4(%rsp), %xmm1

// CHECK: gf2p8affineinvqb  $7, 4(%rsp), %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0xcf,0x4c,0x24,0x04,0x07]
          gf2p8affineinvqb  $7, 4(%rsp), %xmm1

// CHECK: gf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %xmm1
// CHECK: encoding: [0x66,0x42,0x0f,0x3a,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          gf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %xmm1

// CHECK: gf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %xmm1
// CHECK: encoding: [0x66,0x42,0x0f,0x3a,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          gf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %xmm1

// CHECK: gf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %xmm1
// CHECK: encoding: [0x66,0x42,0x0f,0x3a,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          gf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %xmm1

// CHECK: gf2p8affineqb  $7, (%rcx), %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0xce,0x09,0x07]
          gf2p8affineqb  $7, (%rcx), %xmm1

// CHECK: gf2p8affineqb  $7, -4(%rsp), %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0xce,0x4c,0x24,0xfc,0x07]
          gf2p8affineqb  $7, -4(%rsp), %xmm1

// CHECK: gf2p8affineqb  $7, 4(%rsp), %xmm1
// CHECK: encoding: [0x66,0x0f,0x3a,0xce,0x4c,0x24,0x04,0x07]
          gf2p8affineqb  $7, 4(%rsp), %xmm1

// CHECK: gf2p8affineqb  $7, 268435456(%rcx,%r14,8), %xmm1
// CHECK: encoding: [0x66,0x42,0x0f,0x3a,0xce,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          gf2p8affineqb  $7, 268435456(%rcx,%r14,8), %xmm1

// CHECK: gf2p8affineqb  $7, -536870912(%rcx,%r14,8), %xmm1
// CHECK: encoding: [0x66,0x42,0x0f,0x3a,0xce,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          gf2p8affineqb  $7, -536870912(%rcx,%r14,8), %xmm1

// CHECK: gf2p8affineqb  $7, -536870910(%rcx,%r14,8), %xmm1
// CHECK: encoding: [0x66,0x42,0x0f,0x3a,0xce,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          gf2p8affineqb  $7, -536870910(%rcx,%r14,8), %xmm1

// CHECK: gf2p8mulb %xmm2, %xmm1
// CHECK: encoding: [0x66,0x0f,0x38,0xcf,0xca]
          gf2p8mulb %xmm2, %xmm1

// CHECK: gf2p8mulb  (%rcx), %xmm1
// CHECK: encoding: [0x66,0x0f,0x38,0xcf,0x09]
          gf2p8mulb  (%rcx), %xmm1

// CHECK: gf2p8mulb  -4(%rsp), %xmm1
// CHECK: encoding: [0x66,0x0f,0x38,0xcf,0x4c,0x24,0xfc]
          gf2p8mulb  -4(%rsp), %xmm1

// CHECK: gf2p8mulb  4(%rsp), %xmm1
// CHECK: encoding: [0x66,0x0f,0x38,0xcf,0x4c,0x24,0x04]
          gf2p8mulb  4(%rsp), %xmm1

// CHECK: gf2p8mulb  268435456(%rcx,%r14,8), %xmm1
// CHECK: encoding: [0x66,0x42,0x0f,0x38,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          gf2p8mulb  268435456(%rcx,%r14,8), %xmm1

// CHECK: gf2p8mulb  -536870912(%rcx,%r14,8), %xmm1
// CHECK: encoding: [0x66,0x42,0x0f,0x38,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          gf2p8mulb  -536870912(%rcx,%r14,8), %xmm1

// CHECK: gf2p8mulb  -536870910(%rcx,%r14,8), %xmm1
// CHECK: encoding: [0x66,0x42,0x0f,0x38,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          gf2p8mulb  -536870910(%rcx,%r14,8), %xmm1

// CHECK: vgf2p8affineinvqb $7, %xmm2, %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe3,0xa9,0xcf,0xca,0x07]
          vgf2p8affineinvqb $7, %xmm2, %xmm10, %xmm1

// CHECK: vgf2p8affineqb $7, %xmm2, %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe3,0xa9,0xce,0xca,0x07]
          vgf2p8affineqb $7, %xmm2, %xmm10, %xmm1

// CHECK: vgf2p8affineinvqb  $7, (%rcx), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe3,0xa9,0xcf,0x09,0x07]
          vgf2p8affineinvqb  $7, (%rcx), %xmm10, %xmm1

// CHECK: vgf2p8affineinvqb  $7, -4(%rsp), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe3,0xa9,0xcf,0x4c,0x24,0xfc,0x07]
          vgf2p8affineinvqb  $7, -4(%rsp), %xmm10, %xmm1

// CHECK: vgf2p8affineinvqb  $7, 4(%rsp), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe3,0xa9,0xcf,0x4c,0x24,0x04,0x07]
          vgf2p8affineinvqb  $7, 4(%rsp), %xmm10, %xmm1

// CHECK: vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xa3,0xa9,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %xmm10, %xmm1

// CHECK: vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xa3,0xa9,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %xmm10, %xmm1

// CHECK: vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xa3,0xa9,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %xmm10, %xmm1

// CHECK: vgf2p8affineqb  $7, (%rcx), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe3,0xa9,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx), %xmm10, %xmm1

// CHECK: vgf2p8affineqb  $7, -4(%rsp), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe3,0xa9,0xce,0x4c,0x24,0xfc,0x07]
          vgf2p8affineqb  $7, -4(%rsp), %xmm10, %xmm1

// CHECK: vgf2p8affineqb  $7, 4(%rsp), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe3,0xa9,0xce,0x4c,0x24,0x04,0x07]
          vgf2p8affineqb  $7, 4(%rsp), %xmm10, %xmm1

// CHECK: vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xa3,0xa9,0xce,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %xmm10, %xmm1

// CHECK: vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xa3,0xa9,0xce,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %xmm10, %xmm1

// CHECK: vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xa3,0xa9,0xce,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %xmm10, %xmm1

// CHECK: vgf2p8affineinvqb $7, %ymm2, %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe3,0xad,0xcf,0xca,0x07]
          vgf2p8affineinvqb $7, %ymm2, %ymm10, %ymm1

// CHECK: vgf2p8affineqb $7, %ymm2, %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe3,0xad,0xce,0xca,0x07]
          vgf2p8affineqb $7, %ymm2, %ymm10, %ymm1

// CHECK: vgf2p8affineinvqb  $7, (%rcx), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe3,0xad,0xcf,0x09,0x07]
          vgf2p8affineinvqb  $7, (%rcx), %ymm10, %ymm1

// CHECK: vgf2p8affineinvqb  $7, -4(%rsp), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe3,0xad,0xcf,0x4c,0x24,0xfc,0x07]
          vgf2p8affineinvqb  $7, -4(%rsp), %ymm10, %ymm1

// CHECK: vgf2p8affineinvqb  $7, 4(%rsp), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe3,0xad,0xcf,0x4c,0x24,0x04,0x07]
          vgf2p8affineinvqb  $7, 4(%rsp), %ymm10, %ymm1

// CHECK: vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xa3,0xad,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %ymm10, %ymm1

// CHECK: vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xa3,0xad,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %ymm10, %ymm1

// CHECK: vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xa3,0xad,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %ymm10, %ymm1

// CHECK: vgf2p8affineqb  $7, (%rcx), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe3,0xad,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx), %ymm10, %ymm1

// CHECK: vgf2p8affineqb  $7, -4(%rsp), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe3,0xad,0xce,0x4c,0x24,0xfc,0x07]
          vgf2p8affineqb  $7, -4(%rsp), %ymm10, %ymm1

// CHECK: vgf2p8affineqb  $7, 4(%rsp), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe3,0xad,0xce,0x4c,0x24,0x04,0x07]
          vgf2p8affineqb  $7, 4(%rsp), %ymm10, %ymm1

// CHECK: vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xa3,0xad,0xce,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %ymm10, %ymm1

// CHECK: vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xa3,0xad,0xce,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %ymm10, %ymm1

// CHECK: vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xa3,0xad,0xce,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %ymm10, %ymm1

// CHECK: vgf2p8mulb %xmm2, %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x29,0xcf,0xca]
          vgf2p8mulb %xmm2, %xmm10, %xmm1

// CHECK: vgf2p8mulb  (%rcx), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x29,0xcf,0x09]
          vgf2p8mulb  (%rcx), %xmm10, %xmm1

// CHECK: vgf2p8mulb  -4(%rsp), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x29,0xcf,0x4c,0x24,0xfc]
          vgf2p8mulb  -4(%rsp), %xmm10, %xmm1

// CHECK: vgf2p8mulb  4(%rsp), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xe2,0x29,0xcf,0x4c,0x24,0x04]
          vgf2p8mulb  4(%rsp), %xmm10, %xmm1

// CHECK: vgf2p8mulb  268435456(%rcx,%r14,8), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xa2,0x29,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vgf2p8mulb  268435456(%rcx,%r14,8), %xmm10, %xmm1

// CHECK: vgf2p8mulb  -536870912(%rcx,%r14,8), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xa2,0x29,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vgf2p8mulb  -536870912(%rcx,%r14,8), %xmm10, %xmm1

// CHECK: vgf2p8mulb  -536870910(%rcx,%r14,8), %xmm10, %xmm1
// CHECK: encoding: [0xc4,0xa2,0x29,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vgf2p8mulb  -536870910(%rcx,%r14,8), %xmm10, %xmm1

// CHECK: vgf2p8mulb %ymm2, %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x2d,0xcf,0xca]
          vgf2p8mulb %ymm2, %ymm10, %ymm1

// CHECK: vgf2p8mulb  (%rcx), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x2d,0xcf,0x09]
          vgf2p8mulb  (%rcx), %ymm10, %ymm1

// CHECK: vgf2p8mulb  -4(%rsp), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x2d,0xcf,0x4c,0x24,0xfc]
          vgf2p8mulb  -4(%rsp), %ymm10, %ymm1

// CHECK: vgf2p8mulb  4(%rsp), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xe2,0x2d,0xcf,0x4c,0x24,0x04]
          vgf2p8mulb  4(%rsp), %ymm10, %ymm1

// CHECK: vgf2p8mulb  268435456(%rcx,%r14,8), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x2d,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vgf2p8mulb  268435456(%rcx,%r14,8), %ymm10, %ymm1

// CHECK: vgf2p8mulb  -536870912(%rcx,%r14,8), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x2d,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vgf2p8mulb  -536870912(%rcx,%r14,8), %ymm10, %ymm1

// CHECK: vgf2p8mulb  -536870910(%rcx,%r14,8), %ymm10, %ymm1
// CHECK: encoding: [0xc4,0xa2,0x2d,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vgf2p8mulb  -536870910(%rcx,%r14,8), %ymm10, %ymm1

