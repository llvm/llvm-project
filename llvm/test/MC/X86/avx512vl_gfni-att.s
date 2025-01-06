// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s | FileCheck %s

// CHECK: vgf2p8affineinvqb $7, %xmm2, %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x00,0xcf,0xca,0x07]
          vgf2p8affineinvqb $7, %xmm2, %xmm20, %xmm1

// CHECK: vgf2p8affineqb $7, %xmm2, %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x00,0xce,0xca,0x07]
          vgf2p8affineqb $7, %xmm2, %xmm20, %xmm1

// CHECK: vgf2p8affineinvqb $7, %xmm2, %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x02,0xcf,0xca,0x07]
          vgf2p8affineinvqb $7, %xmm2, %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineqb $7, %xmm2, %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x02,0xce,0xca,0x07]
          vgf2p8affineqb $7, %xmm2, %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, (%rcx), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x00,0xcf,0x09,0x07]
          vgf2p8affineinvqb  $7, (%rcx), %xmm20, %xmm1

// CHECK: vgf2p8affineinvqb  $7, -64(%rsp), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x00,0xcf,0x4c,0x24,0xfc,0x07]
          vgf2p8affineinvqb  $7, -64(%rsp), %xmm20, %xmm1

// CHECK: vgf2p8affineinvqb  $7, 64(%rsp), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x00,0xcf,0x4c,0x24,0x04,0x07]
          vgf2p8affineinvqb  $7, 64(%rsp), %xmm20, %xmm1

// CHECK: vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x00,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %xmm20, %xmm1

// CHECK: vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x00,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %xmm20, %xmm1

// CHECK: vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x00,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %xmm20, %xmm1

// CHECK: vgf2p8affineqb  $7, (%rcx), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x00,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx), %xmm20, %xmm1

// CHECK: vgf2p8affineqb  $7, -64(%rsp), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x00,0xce,0x4c,0x24,0xfc,0x07]
          vgf2p8affineqb  $7, -64(%rsp), %xmm20, %xmm1

// CHECK: vgf2p8affineqb  $7, 64(%rsp), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x00,0xce,0x4c,0x24,0x04,0x07]
          vgf2p8affineqb  $7, 64(%rsp), %xmm20, %xmm1

// CHECK: vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x00,0xce,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %xmm20, %xmm1

// CHECK: vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x00,0xce,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %xmm20, %xmm1

// CHECK: vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x00,0xce,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %xmm20, %xmm1

// CHECK: vgf2p8affineinvqb  $7, (%rcx), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x02,0xcf,0x09,0x07]
          vgf2p8affineinvqb  $7, (%rcx), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, -64(%rsp), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x02,0xcf,0x4c,0x24,0xfc,0x07]
          vgf2p8affineinvqb  $7, -64(%rsp), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, 64(%rsp), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x02,0xcf,0x4c,0x24,0x04,0x07]
          vgf2p8affineinvqb  $7, 64(%rsp), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x02,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x02,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x02,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, (%rcx), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x02,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, -64(%rsp), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x02,0xce,0x4c,0x24,0xfc,0x07]
          vgf2p8affineqb  $7, -64(%rsp), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, 64(%rsp), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x02,0xce,0x4c,0x24,0x04,0x07]
          vgf2p8affineqb  $7, 64(%rsp), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x02,0xce,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x02,0xce,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x02,0xce,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8affineinvqb $7, %ymm2, %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x20,0xcf,0xca,0x07]
          vgf2p8affineinvqb $7, %ymm2, %ymm20, %ymm1

// CHECK: vgf2p8affineqb $7, %ymm2, %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x20,0xce,0xca,0x07]
          vgf2p8affineqb $7, %ymm2, %ymm20, %ymm1

// CHECK: vgf2p8affineinvqb $7, %ymm2, %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x22,0xcf,0xca,0x07]
          vgf2p8affineinvqb $7, %ymm2, %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineqb $7, %ymm2, %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x22,0xce,0xca,0x07]
          vgf2p8affineqb $7, %ymm2, %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, (%rcx), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x20,0xcf,0x09,0x07]
          vgf2p8affineinvqb  $7, (%rcx), %ymm20, %ymm1

// CHECK: vgf2p8affineinvqb  $7, -128(%rsp), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x20,0xcf,0x4c,0x24,0xfc,0x07]
          vgf2p8affineinvqb  $7, -128(%rsp), %ymm20, %ymm1

// CHECK: vgf2p8affineinvqb  $7, 128(%rsp), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x20,0xcf,0x4c,0x24,0x04,0x07]
          vgf2p8affineinvqb  $7, 128(%rsp), %ymm20, %ymm1

// CHECK: vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x20,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %ymm20, %ymm1

// CHECK: vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x20,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %ymm20, %ymm1

// CHECK: vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x20,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %ymm20, %ymm1

// CHECK: vgf2p8affineqb  $7, (%rcx), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x20,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx), %ymm20, %ymm1

// CHECK: vgf2p8affineqb  $7, -128(%rsp), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x20,0xce,0x4c,0x24,0xfc,0x07]
          vgf2p8affineqb  $7, -128(%rsp), %ymm20, %ymm1

// CHECK: vgf2p8affineqb  $7, 128(%rsp), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x20,0xce,0x4c,0x24,0x04,0x07]
          vgf2p8affineqb  $7, 128(%rsp), %ymm20, %ymm1

// CHECK: vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x20,0xce,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %ymm20, %ymm1

// CHECK: vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x20,0xce,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %ymm20, %ymm1

// CHECK: vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x20,0xce,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %ymm20, %ymm1

// CHECK: vgf2p8affineinvqb  $7, (%rcx), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x22,0xcf,0x09,0x07]
          vgf2p8affineinvqb  $7, (%rcx), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, -128(%rsp), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x22,0xcf,0x4c,0x24,0xfc,0x07]
          vgf2p8affineinvqb  $7, -128(%rsp), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, 128(%rsp), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x22,0xcf,0x4c,0x24,0x04,0x07]
          vgf2p8affineinvqb  $7, 128(%rsp), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x22,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x22,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x22,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineqb  $7, (%rcx), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x22,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineqb  $7, -128(%rsp), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x22,0xce,0x4c,0x24,0xfc,0x07]
          vgf2p8affineqb  $7, -128(%rsp), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineqb  $7, 128(%rsp), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x22,0xce,0x4c,0x24,0x04,0x07]
          vgf2p8affineqb  $7, 128(%rsp), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x22,0xce,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x22,0xce,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x22,0xce,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8mulb %xmm2, %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x00,0xcf,0xca]
          vgf2p8mulb %xmm2, %xmm20, %xmm1

// CHECK: vgf2p8mulb %xmm2, %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x02,0xcf,0xca]
          vgf2p8mulb %xmm2, %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8mulb  (%rcx), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x00,0xcf,0x09]
          vgf2p8mulb  (%rcx), %xmm20, %xmm1

// CHECK: vgf2p8mulb  -64(%rsp), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x00,0xcf,0x4c,0x24,0xfc]
          vgf2p8mulb  -64(%rsp), %xmm20, %xmm1

// CHECK: vgf2p8mulb  64(%rsp), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x00,0xcf,0x4c,0x24,0x04]
          vgf2p8mulb  64(%rsp), %xmm20, %xmm1

// CHECK: vgf2p8mulb  268435456(%rcx,%r14,8), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xb2,0x5d,0x00,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vgf2p8mulb  268435456(%rcx,%r14,8), %xmm20, %xmm1

// CHECK: vgf2p8mulb  -536870912(%rcx,%r14,8), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xb2,0x5d,0x00,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vgf2p8mulb  -536870912(%rcx,%r14,8), %xmm20, %xmm1

// CHECK: vgf2p8mulb  -536870910(%rcx,%r14,8), %xmm20, %xmm1
// CHECK: encoding: [0x62,0xb2,0x5d,0x00,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vgf2p8mulb  -536870910(%rcx,%r14,8), %xmm20, %xmm1

// CHECK: vgf2p8mulb  (%rcx), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x02,0xcf,0x09]
          vgf2p8mulb  (%rcx), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8mulb  -64(%rsp), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x02,0xcf,0x4c,0x24,0xfc]
          vgf2p8mulb  -64(%rsp), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8mulb  64(%rsp), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x02,0xcf,0x4c,0x24,0x04]
          vgf2p8mulb  64(%rsp), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8mulb  268435456(%rcx,%r14,8), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xb2,0x5d,0x02,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vgf2p8mulb  268435456(%rcx,%r14,8), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8mulb  -536870912(%rcx,%r14,8), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xb2,0x5d,0x02,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vgf2p8mulb  -536870912(%rcx,%r14,8), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8mulb  -536870910(%rcx,%r14,8), %xmm20, %xmm1 {%k2}
// CHECK: encoding: [0x62,0xb2,0x5d,0x02,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vgf2p8mulb  -536870910(%rcx,%r14,8), %xmm20, %xmm1 {%k2}

// CHECK: vgf2p8mulb %ymm2, %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x20,0xcf,0xca]
          vgf2p8mulb %ymm2, %ymm20, %ymm1

// CHECK: vgf2p8mulb %ymm2, %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x22,0xcf,0xca]
          vgf2p8mulb %ymm2, %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8mulb  (%rcx), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x20,0xcf,0x09]
          vgf2p8mulb  (%rcx), %ymm20, %ymm1

// CHECK: vgf2p8mulb  -128(%rsp), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x20,0xcf,0x4c,0x24,0xfc]
          vgf2p8mulb  -128(%rsp), %ymm20, %ymm1

// CHECK: vgf2p8mulb  128(%rsp), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x20,0xcf,0x4c,0x24,0x04]
          vgf2p8mulb  128(%rsp), %ymm20, %ymm1

// CHECK: vgf2p8mulb  268435456(%rcx,%r14,8), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xb2,0x5d,0x20,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vgf2p8mulb  268435456(%rcx,%r14,8), %ymm20, %ymm1

// CHECK: vgf2p8mulb  -536870912(%rcx,%r14,8), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xb2,0x5d,0x20,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vgf2p8mulb  -536870912(%rcx,%r14,8), %ymm20, %ymm1

// CHECK: vgf2p8mulb  -536870910(%rcx,%r14,8), %ymm20, %ymm1
// CHECK: encoding: [0x62,0xb2,0x5d,0x20,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vgf2p8mulb  -536870910(%rcx,%r14,8), %ymm20, %ymm1

// CHECK: vgf2p8mulb  (%rcx), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x22,0xcf,0x09]
          vgf2p8mulb  (%rcx), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8mulb  -128(%rsp), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x22,0xcf,0x4c,0x24,0xfc]
          vgf2p8mulb  -128(%rsp), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8mulb  128(%rsp), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x22,0xcf,0x4c,0x24,0x04]
          vgf2p8mulb  128(%rsp), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8mulb  268435456(%rcx,%r14,8), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xb2,0x5d,0x22,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vgf2p8mulb  268435456(%rcx,%r14,8), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8mulb  -536870912(%rcx,%r14,8), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xb2,0x5d,0x22,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vgf2p8mulb  -536870912(%rcx,%r14,8), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8mulb  -536870910(%rcx,%r14,8), %ymm20, %ymm1 {%k2}
// CHECK: encoding: [0x62,0xb2,0x5d,0x22,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vgf2p8mulb  -536870910(%rcx,%r14,8), %ymm20, %ymm1 {%k2}

// CHECK: vgf2p8affineinvqb $7, (%rcx){1to2}, %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x10,0xcf,0x09,0x07]
          vgf2p8affineinvqb $7, (%rcx){1to2}, %xmm20, %xmm1

// CHECK: vgf2p8affineinvqb $7, (%rcx){1to4}, %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x30,0xcf,0x09,0x07]
          vgf2p8affineinvqb $7, (%rcx){1to4}, %ymm20, %ymm1

// CHECK: vgf2p8affineqb  $7, (%rcx){1to2}, %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x10,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx){1to2}, %xmm20, %xmm1

// CHECK: vgf2p8affineqb  $7, (%rcx){1to4}, %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x30,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx){1to4}, %ymm20, %ymm1

// CHECK: vgf2p8affineinvqb $7, 8(%rcx){1to2}, %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x10,0xcf,0x49,0x01,0x07]
          vgf2p8affineinvqb $7, 8(%rcx){1to2}, %xmm20, %xmm1

// CHECK: vgf2p8affineinvqb $7, 8(%rcx){1to4}, %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x30,0xcf,0x49,0x01,0x07]
          vgf2p8affineinvqb $7, 8(%rcx){1to4}, %ymm20, %ymm1

// CHECK: vgf2p8affineqb  $7, 8(%rcx){1to2}, %xmm20, %xmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x10,0xce,0x49,0x01,0x07]
          vgf2p8affineqb  $7, 8(%rcx){1to2}, %xmm20, %xmm1

// CHECK: vgf2p8affineqb  $7, 8(%rcx){1to4}, %ymm20, %ymm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x30,0xce,0x49,0x01,0x07]
          vgf2p8affineqb  $7, 8(%rcx){1to4}, %ymm20, %ymm1
