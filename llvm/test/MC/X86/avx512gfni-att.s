// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s | FileCheck %s

// CHECK: vgf2p8affineinvqb $7, %zmm2, %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x40,0xcf,0xca,0x07]
          vgf2p8affineinvqb $7, %zmm2, %zmm20, %zmm1

// CHECK: vgf2p8affineqb $7, %zmm2, %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x40,0xce,0xca,0x07]
          vgf2p8affineqb $7, %zmm2, %zmm20, %zmm1

// CHECK: vgf2p8affineinvqb $7, %zmm2, %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x42,0xcf,0xca,0x07]
          vgf2p8affineinvqb $7, %zmm2, %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineqb $7, %zmm2, %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x42,0xce,0xca,0x07]
          vgf2p8affineqb $7, %zmm2, %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, (%rcx), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x40,0xcf,0x09,0x07]
          vgf2p8affineinvqb  $7, (%rcx), %zmm20, %zmm1

// CHECK: vgf2p8affineinvqb  $7, -256(%rsp), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x40,0xcf,0x4c,0x24,0xfc,0x07]
          vgf2p8affineinvqb  $7, -256(%rsp), %zmm20, %zmm1

// CHECK: vgf2p8affineinvqb  $7, 256(%rsp), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x40,0xcf,0x4c,0x24,0x04,0x07]
          vgf2p8affineinvqb  $7, 256(%rsp), %zmm20, %zmm1

// CHECK: vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x40,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %zmm20, %zmm1

// CHECK: vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x40,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %zmm20, %zmm1

// CHECK: vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x40,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %zmm20, %zmm1

// CHECK: vgf2p8affineqb  $7, (%rcx), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x40,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx), %zmm20, %zmm1

// CHECK: vgf2p8affineqb  $7, -256(%rsp), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x40,0xce,0x4c,0x24,0xfc,0x07]
          vgf2p8affineqb  $7, -256(%rsp), %zmm20, %zmm1

// CHECK: vgf2p8affineqb  $7, 256(%rsp), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x40,0xce,0x4c,0x24,0x04,0x07]
          vgf2p8affineqb  $7, 256(%rsp), %zmm20, %zmm1

// CHECK: vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x40,0xce,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %zmm20, %zmm1

// CHECK: vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x40,0xce,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %zmm20, %zmm1

// CHECK: vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xb3,0xdd,0x40,0xce,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %zmm20, %zmm1

// CHECK: vgf2p8affineinvqb  $7, (%rcx), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x42,0xcf,0x09,0x07]
          vgf2p8affineinvqb  $7, (%rcx), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, -256(%rsp), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x42,0xcf,0x4c,0x24,0xfc,0x07]
          vgf2p8affineinvqb  $7, -256(%rsp), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, 256(%rsp), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x42,0xcf,0x4c,0x24,0x04,0x07]
          vgf2p8affineinvqb  $7, 256(%rsp), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x42,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineinvqb  $7, 268435456(%rcx,%r14,8), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x42,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870912(%rcx,%r14,8), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x42,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineinvqb  $7, -536870910(%rcx,%r14,8), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, (%rcx), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x42,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, -256(%rsp), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x42,0xce,0x4c,0x24,0xfc,0x07]
          vgf2p8affineqb  $7, -256(%rsp), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, 256(%rsp), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf3,0xdd,0x42,0xce,0x4c,0x24,0x04,0x07]
          vgf2p8affineqb  $7, 256(%rsp), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x42,0xce,0x8c,0xf1,0x00,0x00,0x00,0x10,0x07]
          vgf2p8affineqb  $7, 268435456(%rcx,%r14,8), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x42,0xce,0x8c,0xf1,0x00,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870912(%rcx,%r14,8), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xb3,0xdd,0x42,0xce,0x8c,0xf1,0x02,0x00,0x00,0xe0,0x07]
          vgf2p8affineqb  $7, -536870910(%rcx,%r14,8), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8mulb %zmm2, %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x40,0xcf,0xca]
          vgf2p8mulb %zmm2, %zmm20, %zmm1

// CHECK: vgf2p8mulb %zmm2, %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x42,0xcf,0xca]
          vgf2p8mulb %zmm2, %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8mulb  (%rcx), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x40,0xcf,0x09]
          vgf2p8mulb  (%rcx), %zmm20, %zmm1

// CHECK: vgf2p8mulb  -256(%rsp), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x40,0xcf,0x4c,0x24,0xfc]
          vgf2p8mulb  -256(%rsp), %zmm20, %zmm1

// CHECK: vgf2p8mulb  256(%rsp), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf2,0x5d,0x40,0xcf,0x4c,0x24,0x04]
          vgf2p8mulb  256(%rsp), %zmm20, %zmm1

// CHECK: vgf2p8mulb  268435456(%rcx,%r14,8), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xb2,0x5d,0x40,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vgf2p8mulb  268435456(%rcx,%r14,8), %zmm20, %zmm1

// CHECK: vgf2p8mulb  -536870912(%rcx,%r14,8), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xb2,0x5d,0x40,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vgf2p8mulb  -536870912(%rcx,%r14,8), %zmm20, %zmm1

// CHECK: vgf2p8mulb  -536870910(%rcx,%r14,8), %zmm20, %zmm1
// CHECK: encoding: [0x62,0xb2,0x5d,0x40,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vgf2p8mulb  -536870910(%rcx,%r14,8), %zmm20, %zmm1

// CHECK: vgf2p8mulb  (%rcx), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x42,0xcf,0x09]
          vgf2p8mulb  (%rcx), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8mulb  -256(%rsp), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x42,0xcf,0x4c,0x24,0xfc]
          vgf2p8mulb  -256(%rsp), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8mulb  256(%rsp), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xf2,0x5d,0x42,0xcf,0x4c,0x24,0x04]
          vgf2p8mulb  256(%rsp), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8mulb  268435456(%rcx,%r14,8), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xb2,0x5d,0x42,0xcf,0x8c,0xf1,0x00,0x00,0x00,0x10]
          vgf2p8mulb  268435456(%rcx,%r14,8), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8mulb  -536870912(%rcx,%r14,8), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xb2,0x5d,0x42,0xcf,0x8c,0xf1,0x00,0x00,0x00,0xe0]
          vgf2p8mulb  -536870912(%rcx,%r14,8), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8mulb  -536870910(%rcx,%r14,8), %zmm20, %zmm1 {%k2}
// CHECK: encoding: [0x62,0xb2,0x5d,0x42,0xcf,0x8c,0xf1,0x02,0x00,0x00,0xe0]
          vgf2p8mulb  -536870910(%rcx,%r14,8), %zmm20, %zmm1 {%k2}

// CHECK: vgf2p8affineinvqb $7, (%rcx){1to8}, %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x50,0xcf,0x09,0x07]
          vgf2p8affineinvqb $7, (%rcx){1to8}, %zmm20, %zmm1

// CHECK: vgf2p8affineqb  $7, (%rcx){1to8}, %zmm20, %zmm1
// CHECK: encoding: [0x62,0xf3,0xdd,0x50,0xce,0x09,0x07]
          vgf2p8affineqb  $7, (%rcx){1to8}, %zmm20, %zmm1

