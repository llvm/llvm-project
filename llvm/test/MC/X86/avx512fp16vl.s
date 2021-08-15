// RUN: llvm-mc -triple i686-unknown-unknown --show-encoding < %s  | FileCheck %s

// CHECK: vaddph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x58,0xf4]
          vaddph %ymm4, %ymm5, %ymm6

// CHECK: vaddph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x58,0xf4]
          vaddph %xmm4, %xmm5, %xmm6

// CHECK: vaddph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x2f,0x58,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vaddph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vaddph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x38,0x58,0x31]
          vaddph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vaddph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x58,0x71,0x7f]
          vaddph  4064(%ecx), %ymm5, %ymm6

// CHECK: vaddph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0xbf,0x58,0x72,0x80]
          vaddph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vaddph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x0f,0x58,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vaddph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vaddph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x58,0x31]
          vaddph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vaddph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x58,0x71,0x7f]
          vaddph  2032(%ecx), %xmm5, %xmm6

// CHECK: vaddph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0x9f,0x58,0x72,0x80]
          vaddph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vcmpeqph %ymm4, %ymm5, %k5
// CHECK: encoding: [0x62,0xf3,0x54,0x28,0xc2,0xec,0x00]
          vcmpph $0, %ymm4, %ymm5, %k5

// CHECK: vcmpltph %xmm4, %xmm5, %k5
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0xc2,0xec,0x01]
          vcmpph $1, %xmm4, %xmm5, %k5

// CHECK: vcmpleph 268435456(%esp,%esi,8), %xmm5, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x54,0x0f,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x02]
          vcmpph  $2, 268435456(%esp,%esi,8), %xmm5, %k5 {%k7}

// CHECK: vcmpunordph (%ecx){1to8}, %xmm5, %k5
// CHECK: encoding: [0x62,0xf3,0x54,0x18,0xc2,0x29,0x03]
          vcmpph  $3, (%ecx){1to8}, %xmm5, %k5

// CHECK: vcmpneqph 2032(%ecx), %xmm5, %k5
// CHECK: encoding: [0x62,0xf3,0x54,0x08,0xc2,0x69,0x7f,0x04]
          vcmpph  $4, 2032(%ecx), %xmm5, %k5

// CHECK: vcmpnltph -256(%edx){1to8}, %xmm5, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x54,0x1f,0xc2,0x6a,0x80,0x05]
          vcmpph  $5, -256(%edx){1to8}, %xmm5, %k5 {%k7}

// CHECK: vcmpnleph 268435456(%esp,%esi,8), %ymm5, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x54,0x2f,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x06]
          vcmpph  $6, 268435456(%esp,%esi,8), %ymm5, %k5 {%k7}

// CHECK: vcmpordph (%ecx){1to16}, %ymm5, %k5
// CHECK: encoding: [0x62,0xf3,0x54,0x38,0xc2,0x29,0x07]
          vcmpph  $7, (%ecx){1to16}, %ymm5, %k5

// CHECK: vcmpeq_uqph 4064(%ecx), %ymm5, %k5
// CHECK: encoding: [0x62,0xf3,0x54,0x28,0xc2,0x69,0x7f,0x08]
          vcmpph  $8, 4064(%ecx), %ymm5, %k5

// CHECK: vcmpngeph -256(%edx){1to16}, %ymm5, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x54,0x3f,0xc2,0x6a,0x80,0x09]
          vcmpph  $9, -256(%edx){1to16}, %ymm5, %k5 {%k7}

// CHECK: vdivph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x5e,0xf4]
          vdivph %ymm4, %ymm5, %ymm6

// CHECK: vdivph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x5e,0xf4]
          vdivph %xmm4, %xmm5, %xmm6

// CHECK: vdivph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x2f,0x5e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdivph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vdivph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x38,0x5e,0x31]
          vdivph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vdivph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x5e,0x71,0x7f]
          vdivph  4064(%ecx), %ymm5, %ymm6

// CHECK: vdivph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0xbf,0x5e,0x72,0x80]
          vdivph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vdivph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x0f,0x5e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdivph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vdivph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x5e,0x31]
          vdivph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vdivph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x5e,0x71,0x7f]
          vdivph  2032(%ecx), %xmm5, %xmm6

// CHECK: vdivph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0x9f,0x5e,0x72,0x80]
          vdivph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vmaxph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x5f,0xf4]
          vmaxph %ymm4, %ymm5, %ymm6

// CHECK: vmaxph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x5f,0xf4]
          vmaxph %xmm4, %xmm5, %xmm6

// CHECK: vmaxph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x2f,0x5f,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmaxph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vmaxph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x38,0x5f,0x31]
          vmaxph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vmaxph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x5f,0x71,0x7f]
          vmaxph  4064(%ecx), %ymm5, %ymm6

// CHECK: vmaxph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0xbf,0x5f,0x72,0x80]
          vmaxph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vmaxph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x0f,0x5f,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmaxph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vmaxph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x5f,0x31]
          vmaxph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vmaxph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x5f,0x71,0x7f]
          vmaxph  2032(%ecx), %xmm5, %xmm6

// CHECK: vmaxph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0x9f,0x5f,0x72,0x80]
          vmaxph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vminph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x5d,0xf4]
          vminph %ymm4, %ymm5, %ymm6

// CHECK: vminph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x5d,0xf4]
          vminph %xmm4, %xmm5, %xmm6

// CHECK: vminph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x2f,0x5d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vminph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vminph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x38,0x5d,0x31]
          vminph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vminph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x5d,0x71,0x7f]
          vminph  4064(%ecx), %ymm5, %ymm6

// CHECK: vminph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0xbf,0x5d,0x72,0x80]
          vminph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vminph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x0f,0x5d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vminph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vminph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x5d,0x31]
          vminph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vminph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x5d,0x71,0x7f]
          vminph  2032(%ecx), %xmm5, %xmm6

// CHECK: vminph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0x9f,0x5d,0x72,0x80]
          vminph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vmulph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x59,0xf4]
          vmulph %ymm4, %ymm5, %ymm6

// CHECK: vmulph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x59,0xf4]
          vmulph %xmm4, %xmm5, %xmm6

// CHECK: vmulph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x2f,0x59,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmulph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vmulph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x38,0x59,0x31]
          vmulph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vmulph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x59,0x71,0x7f]
          vmulph  4064(%ecx), %ymm5, %ymm6

// CHECK: vmulph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0xbf,0x59,0x72,0x80]
          vmulph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vmulph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x0f,0x59,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vmulph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vmulph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x59,0x31]
          vmulph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vmulph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x59,0x71,0x7f]
          vmulph  2032(%ecx), %xmm5, %xmm6

// CHECK: vmulph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0x9f,0x59,0x72,0x80]
          vmulph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vsubph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x5c,0xf4]
          vsubph %ymm4, %ymm5, %ymm6

// CHECK: vsubph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x5c,0xf4]
          vsubph %xmm4, %xmm5, %xmm6

// CHECK: vsubph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x2f,0x5c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vsubph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vsubph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x38,0x5c,0x31]
          vsubph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vsubph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x54,0x28,0x5c,0x71,0x7f]
          vsubph  4064(%ecx), %ymm5, %ymm6

// CHECK: vsubph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0xbf,0x5c,0x72,0x80]
          vsubph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vsubph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x54,0x0f,0x5c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vsubph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vsubph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x18,0x5c,0x31]
          vsubph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vsubph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x54,0x08,0x5c,0x71,0x7f]
          vsubph  2032(%ecx), %xmm5, %xmm6

// CHECK: vsubph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x54,0x9f,0x5c,0x72,0x80]
          vsubph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
