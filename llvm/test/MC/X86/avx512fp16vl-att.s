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

// CHECK: vcvtdq2ph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x5b,0xf5]
          vcvtdq2ph %xmm5, %xmm6

// CHECK: vcvtdq2ph %ymm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x5b,0xf5]
          vcvtdq2ph %ymm5, %xmm6

// CHECK: vcvtdq2phx  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtdq2phx  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtdq2ph  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x5b,0x31]
          vcvtdq2ph  (%ecx){1to4}, %xmm6

// CHECK: vcvtdq2phx  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x5b,0x71,0x7f]
          vcvtdq2phx  2032(%ecx), %xmm6

// CHECK: vcvtdq2ph  -512(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x5b,0x72,0x80]
          vcvtdq2ph  -512(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvtdq2ph  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x5b,0x31]
          vcvtdq2ph  (%ecx){1to8}, %xmm6

// CHECK: vcvtdq2phy  4064(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x5b,0x71,0x7f]
          vcvtdq2phy  4064(%ecx), %xmm6

// CHECK: vcvtdq2ph  -512(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x5b,0x72,0x80]
          vcvtdq2ph  -512(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vcvtpd2ph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x5a,0xf5]
          vcvtpd2ph %xmm5, %xmm6

// CHECK: vcvtpd2ph %ymm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x5a,0xf5]
          vcvtpd2ph %ymm5, %xmm6

// CHECK: vcvtpd2phx  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfd,0x0f,0x5a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtpd2phx  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtpd2ph  (%ecx){1to2}, %xmm6
// CHECK: encoding: [0x62,0xf5,0xfd,0x18,0x5a,0x31]
          vcvtpd2ph  (%ecx){1to2}, %xmm6

// CHECK: vcvtpd2phx  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0xfd,0x08,0x5a,0x71,0x7f]
          vcvtpd2phx  2032(%ecx), %xmm6

// CHECK: vcvtpd2ph  -1024(%edx){1to2}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0x9f,0x5a,0x72,0x80]
          vcvtpd2ph  -1024(%edx){1to2}, %xmm6 {%k7} {z}

// CHECK: vcvtpd2ph  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0xfd,0x38,0x5a,0x31]
          vcvtpd2ph  (%ecx){1to4}, %xmm6

// CHECK: vcvtpd2phy  4064(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0xfd,0x28,0x5a,0x71,0x7f]
          vcvtpd2phy  4064(%ecx), %xmm6

// CHECK: vcvtpd2ph  -1024(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfd,0xbf,0x5a,0x72,0x80]
          vcvtpd2ph  -1024(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvtph2dq %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x5b,0xf5]
          vcvtph2dq %xmm5, %xmm6

// CHECK: vcvtph2dq %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x5b,0xf5]
          vcvtph2dq %xmm5, %ymm6

// CHECK: vcvtph2dq  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2dq  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtph2dq  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x5b,0x31]
          vcvtph2dq  (%ecx){1to4}, %xmm6

// CHECK: vcvtph2dq  1016(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x5b,0x71,0x7f]
          vcvtph2dq  1016(%ecx), %xmm6

// CHECK: vcvtph2dq  -256(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x5b,0x72,0x80]
          vcvtph2dq  -256(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvtph2dq  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2dq  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtph2dq  (%ecx){1to8}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x5b,0x31]
          vcvtph2dq  (%ecx){1to8}, %ymm6

// CHECK: vcvtph2dq  2032(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x5b,0x71,0x7f]
          vcvtph2dq  2032(%ecx), %ymm6

// CHECK: vcvtph2dq  -256(%edx){1to8}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x5b,0x72,0x80]
          vcvtph2dq  -256(%edx){1to8}, %ymm6 {%k7} {z}

// CHECK: vcvtph2pd %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x5a,0xf5]
          vcvtph2pd %xmm5, %xmm6

// CHECK: vcvtph2pd %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x5a,0xf5]
          vcvtph2pd %xmm5, %ymm6

// CHECK: vcvtph2pd  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x5a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2pd  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtph2pd  (%ecx){1to2}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x5a,0x31]
          vcvtph2pd  (%ecx){1to2}, %xmm6

// CHECK: vcvtph2pd  508(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x5a,0x71,0x7f]
          vcvtph2pd  508(%ecx), %xmm6

// CHECK: vcvtph2pd  -256(%edx){1to2}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x5a,0x72,0x80]
          vcvtph2pd  -256(%edx){1to2}, %xmm6 {%k7} {z}

// CHECK: vcvtph2pd  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x5a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2pd  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtph2pd  (%ecx){1to4}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x5a,0x31]
          vcvtph2pd  (%ecx){1to4}, %ymm6

// CHECK: vcvtph2pd  1016(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x5a,0x71,0x7f]
          vcvtph2pd  1016(%ecx), %ymm6

// CHECK: vcvtph2pd  -256(%edx){1to4}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x5a,0x72,0x80]
          vcvtph2pd  -256(%edx){1to4}, %ymm6 {%k7} {z}

// CHECK: vcvtph2psx %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x08,0x13,0xf5]
          vcvtph2psx %xmm5, %xmm6

// CHECK: vcvtph2psx %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x28,0x13,0xf5]
          vcvtph2psx %xmm5, %ymm6

// CHECK: vcvtph2psx  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7d,0x0f,0x13,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2psx  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtph2psx  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x18,0x13,0x31]
          vcvtph2psx  (%ecx){1to4}, %xmm6

// CHECK: vcvtph2psx  1016(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x08,0x13,0x71,0x7f]
          vcvtph2psx  1016(%ecx), %xmm6

// CHECK: vcvtph2psx  -256(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7d,0x9f,0x13,0x72,0x80]
          vcvtph2psx  -256(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvtph2psx  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7d,0x2f,0x13,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2psx  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtph2psx  (%ecx){1to8}, %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x38,0x13,0x31]
          vcvtph2psx  (%ecx){1to8}, %ymm6

// CHECK: vcvtph2psx  2032(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x28,0x13,0x71,0x7f]
          vcvtph2psx  2032(%ecx), %ymm6

// CHECK: vcvtph2psx  -256(%edx){1to8}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7d,0xbf,0x13,0x72,0x80]
          vcvtph2psx  -256(%edx){1to8}, %ymm6 {%k7} {z}

// CHECK: vcvtph2qq %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7b,0xf5]
          vcvtph2qq %xmm5, %xmm6

// CHECK: vcvtph2qq %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x7b,0xf5]
          vcvtph2qq %xmm5, %ymm6

// CHECK: vcvtph2qq  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x7b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2qq  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtph2qq  (%ecx){1to2}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x7b,0x31]
          vcvtph2qq  (%ecx){1to2}, %xmm6

// CHECK: vcvtph2qq  508(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7b,0x71,0x7f]
          vcvtph2qq  508(%ecx), %xmm6

// CHECK: vcvtph2qq  -256(%edx){1to2}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x7b,0x72,0x80]
          vcvtph2qq  -256(%edx){1to2}, %xmm6 {%k7} {z}

// CHECK: vcvtph2qq  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x7b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2qq  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtph2qq  (%ecx){1to4}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x7b,0x31]
          vcvtph2qq  (%ecx){1to4}, %ymm6

// CHECK: vcvtph2qq  1016(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x7b,0x71,0x7f]
          vcvtph2qq  1016(%ecx), %ymm6

// CHECK: vcvtph2qq  -256(%edx){1to4}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x7b,0x72,0x80]
          vcvtph2qq  -256(%edx){1to4}, %ymm6 {%k7} {z}

// CHECK: vcvtph2udq %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x79,0xf5]
          vcvtph2udq %xmm5, %xmm6

// CHECK: vcvtph2udq %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x79,0xf5]
          vcvtph2udq %xmm5, %ymm6

// CHECK: vcvtph2udq  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x79,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2udq  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtph2udq  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x79,0x31]
          vcvtph2udq  (%ecx){1to4}, %xmm6

// CHECK: vcvtph2udq  1016(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x79,0x71,0x7f]
          vcvtph2udq  1016(%ecx), %xmm6

// CHECK: vcvtph2udq  -256(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x79,0x72,0x80]
          vcvtph2udq  -256(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvtph2udq  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x79,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2udq  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtph2udq  (%ecx){1to8}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x79,0x31]
          vcvtph2udq  (%ecx){1to8}, %ymm6

// CHECK: vcvtph2udq  2032(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x79,0x71,0x7f]
          vcvtph2udq  2032(%ecx), %ymm6

// CHECK: vcvtph2udq  -256(%edx){1to8}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x79,0x72,0x80]
          vcvtph2udq  -256(%edx){1to8}, %ymm6 {%k7} {z}

// CHECK: vcvtph2uqq %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x79,0xf5]
          vcvtph2uqq %xmm5, %xmm6

// CHECK: vcvtph2uqq %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x79,0xf5]
          vcvtph2uqq %xmm5, %ymm6

// CHECK: vcvtph2uqq  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x79,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2uqq  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtph2uqq  (%ecx){1to2}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x79,0x31]
          vcvtph2uqq  (%ecx){1to2}, %xmm6

// CHECK: vcvtph2uqq  508(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x79,0x71,0x7f]
          vcvtph2uqq  508(%ecx), %xmm6

// CHECK: vcvtph2uqq  -256(%edx){1to2}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x79,0x72,0x80]
          vcvtph2uqq  -256(%edx){1to2}, %xmm6 {%k7} {z}

// CHECK: vcvtph2uqq  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x79,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2uqq  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtph2uqq  (%ecx){1to4}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x79,0x31]
          vcvtph2uqq  (%ecx){1to4}, %ymm6

// CHECK: vcvtph2uqq  1016(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x79,0x71,0x7f]
          vcvtph2uqq  1016(%ecx), %ymm6

// CHECK: vcvtph2uqq  -256(%edx){1to4}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x79,0x72,0x80]
          vcvtph2uqq  -256(%edx){1to4}, %ymm6 {%k7} {z}

// CHECK: vcvtph2uw %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x7d,0xf5]
          vcvtph2uw %xmm5, %xmm6

// CHECK: vcvtph2uw %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x7d,0xf5]
          vcvtph2uw %ymm5, %ymm6

// CHECK: vcvtph2uw  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2uw  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtph2uw  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x7d,0x31]
          vcvtph2uw  (%ecx){1to8}, %xmm6

// CHECK: vcvtph2uw  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x7d,0x71,0x7f]
          vcvtph2uw  2032(%ecx), %xmm6

// CHECK: vcvtph2uw  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x7d,0x72,0x80]
          vcvtph2uw  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vcvtph2uw  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2uw  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtph2uw  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x7d,0x31]
          vcvtph2uw  (%ecx){1to16}, %ymm6

// CHECK: vcvtph2uw  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x7d,0x71,0x7f]
          vcvtph2uw  4064(%ecx), %ymm6

// CHECK: vcvtph2uw  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x7d,0x72,0x80]
          vcvtph2uw  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vcvtph2w %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7d,0xf5]
          vcvtph2w %xmm5, %xmm6

// CHECK: vcvtph2w %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x7d,0xf5]
          vcvtph2w %ymm5, %ymm6

// CHECK: vcvtph2w  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2w  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtph2w  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x7d,0x31]
          vcvtph2w  (%ecx){1to8}, %xmm6

// CHECK: vcvtph2w  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7d,0x71,0x7f]
          vcvtph2w  2032(%ecx), %xmm6

// CHECK: vcvtph2w  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x7d,0x72,0x80]
          vcvtph2w  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vcvtph2w  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2w  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtph2w  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x7d,0x31]
          vcvtph2w  (%ecx){1to16}, %ymm6

// CHECK: vcvtph2w  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x7d,0x71,0x7f]
          vcvtph2w  4064(%ecx), %ymm6

// CHECK: vcvtph2w  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x7d,0x72,0x80]
          vcvtph2w  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vcvtps2phx %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x1d,0xf5]
          vcvtps2phx %xmm5, %xmm6

// CHECK: vcvtps2phx %ymm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x1d,0xf5]
          vcvtps2phx %ymm5, %xmm6

// CHECK: vcvtps2phxx  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x1d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2phxx  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtps2phx  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x1d,0x31]
          vcvtps2phx  (%ecx){1to4}, %xmm6

// CHECK: vcvtps2phxx  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x1d,0x71,0x7f]
          vcvtps2phxx  2032(%ecx), %xmm6

// CHECK: vcvtps2phx  -512(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x1d,0x72,0x80]
          vcvtps2phx  -512(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvtps2phx  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x1d,0x31]
          vcvtps2phx  (%ecx){1to8}, %xmm6

// CHECK: vcvtps2phxy  4064(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x1d,0x71,0x7f]
          vcvtps2phxy  4064(%ecx), %xmm6

// CHECK: vcvtps2phx  -512(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x1d,0x72,0x80]
          vcvtps2phx  -512(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vcvtqq2ph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x5b,0xf5]
          vcvtqq2ph %xmm5, %xmm6

// CHECK: vcvtqq2ph %ymm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x5b,0xf5]
          vcvtqq2ph %ymm5, %xmm6

// CHECK: vcvtqq2phx  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0xfc,0x0f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtqq2phx  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtqq2ph  (%ecx){1to2}, %xmm6
// CHECK: encoding: [0x62,0xf5,0xfc,0x18,0x5b,0x31]
          vcvtqq2ph  (%ecx){1to2}, %xmm6

// CHECK: vcvtqq2phx  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0xfc,0x08,0x5b,0x71,0x7f]
          vcvtqq2phx  2032(%ecx), %xmm6

// CHECK: vcvtqq2ph  -1024(%edx){1to2}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0x9f,0x5b,0x72,0x80]
          vcvtqq2ph  -1024(%edx){1to2}, %xmm6 {%k7} {z}

// CHECK: vcvtqq2ph  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0xfc,0x38,0x5b,0x31]
          vcvtqq2ph  (%ecx){1to4}, %xmm6

// CHECK: vcvtqq2phy  4064(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0xfc,0x28,0x5b,0x71,0x7f]
          vcvtqq2phy  4064(%ecx), %xmm6

// CHECK: vcvtqq2ph  -1024(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xfc,0xbf,0x5b,0x72,0x80]
          vcvtqq2ph  -1024(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvttph2dq %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x5b,0xf5]
          vcvttph2dq %xmm5, %xmm6

// CHECK: vcvttph2dq %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x5b,0xf5]
          vcvttph2dq %xmm5, %ymm6

// CHECK: vcvttph2dq  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2dq  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvttph2dq  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x5b,0x31]
          vcvttph2dq  (%ecx){1to4}, %xmm6

// CHECK: vcvttph2dq  1016(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x5b,0x71,0x7f]
          vcvttph2dq  1016(%ecx), %xmm6

// CHECK: vcvttph2dq  -256(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x9f,0x5b,0x72,0x80]
          vcvttph2dq  -256(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvttph2dq  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x2f,0x5b,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2dq  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvttph2dq  (%ecx){1to8}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x5b,0x31]
          vcvttph2dq  (%ecx){1to8}, %ymm6

// CHECK: vcvttph2dq  2032(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x5b,0x71,0x7f]
          vcvttph2dq  2032(%ecx), %ymm6

// CHECK: vcvttph2dq  -256(%edx){1to8}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xbf,0x5b,0x72,0x80]
          vcvttph2dq  -256(%edx){1to8}, %ymm6 {%k7} {z}

// CHECK: vcvttph2qq %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7a,0xf5]
          vcvttph2qq %xmm5, %xmm6

// CHECK: vcvttph2qq %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x7a,0xf5]
          vcvttph2qq %xmm5, %ymm6

// CHECK: vcvttph2qq  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x7a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2qq  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvttph2qq  (%ecx){1to2}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x7a,0x31]
          vcvttph2qq  (%ecx){1to2}, %xmm6

// CHECK: vcvttph2qq  508(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7a,0x71,0x7f]
          vcvttph2qq  508(%ecx), %xmm6

// CHECK: vcvttph2qq  -256(%edx){1to2}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x7a,0x72,0x80]
          vcvttph2qq  -256(%edx){1to2}, %xmm6 {%k7} {z}

// CHECK: vcvttph2qq  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x7a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2qq  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvttph2qq  (%ecx){1to4}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x7a,0x31]
          vcvttph2qq  (%ecx){1to4}, %ymm6

// CHECK: vcvttph2qq  1016(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x7a,0x71,0x7f]
          vcvttph2qq  1016(%ecx), %ymm6

// CHECK: vcvttph2qq  -256(%edx){1to4}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x7a,0x72,0x80]
          vcvttph2qq  -256(%edx){1to4}, %ymm6 {%k7} {z}

// CHECK: vcvttph2udq %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x78,0xf5]
          vcvttph2udq %xmm5, %xmm6

// CHECK: vcvttph2udq %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x78,0xf5]
          vcvttph2udq %xmm5, %ymm6

// CHECK: vcvttph2udq  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x78,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2udq  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvttph2udq  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x78,0x31]
          vcvttph2udq  (%ecx){1to4}, %xmm6

// CHECK: vcvttph2udq  1016(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x78,0x71,0x7f]
          vcvttph2udq  1016(%ecx), %xmm6

// CHECK: vcvttph2udq  -256(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x78,0x72,0x80]
          vcvttph2udq  -256(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvttph2udq  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x78,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2udq  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvttph2udq  (%ecx){1to8}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x78,0x31]
          vcvttph2udq  (%ecx){1to8}, %ymm6

// CHECK: vcvttph2udq  2032(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x78,0x71,0x7f]
          vcvttph2udq  2032(%ecx), %ymm6

// CHECK: vcvttph2udq  -256(%edx){1to8}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x78,0x72,0x80]
          vcvttph2udq  -256(%edx){1to8}, %ymm6 {%k7} {z}

// CHECK: vcvttph2uqq %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x78,0xf5]
          vcvttph2uqq %xmm5, %xmm6

// CHECK: vcvttph2uqq %xmm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x78,0xf5]
          vcvttph2uqq %xmm5, %ymm6

// CHECK: vcvttph2uqq  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x78,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2uqq  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvttph2uqq  (%ecx){1to2}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x78,0x31]
          vcvttph2uqq  (%ecx){1to2}, %xmm6

// CHECK: vcvttph2uqq  508(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x78,0x71,0x7f]
          vcvttph2uqq  508(%ecx), %xmm6

// CHECK: vcvttph2uqq  -256(%edx){1to2}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x78,0x72,0x80]
          vcvttph2uqq  -256(%edx){1to2}, %xmm6 {%k7} {z}

// CHECK: vcvttph2uqq  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x78,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2uqq  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvttph2uqq  (%ecx){1to4}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x78,0x31]
          vcvttph2uqq  (%ecx){1to4}, %ymm6

// CHECK: vcvttph2uqq  1016(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x78,0x71,0x7f]
          vcvttph2uqq  1016(%ecx), %ymm6

// CHECK: vcvttph2uqq  -256(%edx){1to4}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x78,0x72,0x80]
          vcvttph2uqq  -256(%edx){1to4}, %ymm6 {%k7} {z}

// CHECK: vcvttph2uw %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x7c,0xf5]
          vcvttph2uw %xmm5, %xmm6

// CHECK: vcvttph2uw %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x7c,0xf5]
          vcvttph2uw %ymm5, %ymm6

// CHECK: vcvttph2uw  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x7c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2uw  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvttph2uw  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x7c,0x31]
          vcvttph2uw  (%ecx){1to8}, %xmm6

// CHECK: vcvttph2uw  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x7c,0x71,0x7f]
          vcvttph2uw  2032(%ecx), %xmm6

// CHECK: vcvttph2uw  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x7c,0x72,0x80]
          vcvttph2uw  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vcvttph2uw  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x7c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2uw  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvttph2uw  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x7c,0x31]
          vcvttph2uw  (%ecx){1to16}, %ymm6

// CHECK: vcvttph2uw  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x7c,0x71,0x7f]
          vcvttph2uw  4064(%ecx), %ymm6

// CHECK: vcvttph2uw  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x7c,0x72,0x80]
          vcvttph2uw  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vcvttph2w %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7c,0xf5]
          vcvttph2w %xmm5, %xmm6

// CHECK: vcvttph2w %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x7c,0xf5]
          vcvttph2w %ymm5, %ymm6

// CHECK: vcvttph2w  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x7c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2w  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvttph2w  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x7c,0x31]
          vcvttph2w  (%ecx){1to8}, %xmm6

// CHECK: vcvttph2w  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x7c,0x71,0x7f]
          vcvttph2w  2032(%ecx), %xmm6

// CHECK: vcvttph2w  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x7c,0x72,0x80]
          vcvttph2w  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vcvttph2w  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x7c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2w  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvttph2w  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x7c,0x31]
          vcvttph2w  (%ecx){1to16}, %ymm6

// CHECK: vcvttph2w  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x7c,0x71,0x7f]
          vcvttph2w  4064(%ecx), %ymm6

// CHECK: vcvttph2w  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x7c,0x72,0x80]
          vcvttph2w  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vcvtudq2ph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x7a,0xf5]
          vcvtudq2ph %xmm5, %xmm6

// CHECK: vcvtudq2ph %ymm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x7a,0xf5]
          vcvtudq2ph %ymm5, %xmm6

// CHECK: vcvtudq2phx  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x7a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtudq2phx  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtudq2ph  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x7a,0x31]
          vcvtudq2ph  (%ecx){1to4}, %xmm6

// CHECK: vcvtudq2phx  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x7a,0x71,0x7f]
          vcvtudq2phx  2032(%ecx), %xmm6

// CHECK: vcvtudq2ph  -512(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x7a,0x72,0x80]
          vcvtudq2ph  -512(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvtudq2ph  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x7a,0x31]
          vcvtudq2ph  (%ecx){1to8}, %xmm6

// CHECK: vcvtudq2phy  4064(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x7a,0x71,0x7f]
          vcvtudq2phy  4064(%ecx), %xmm6

// CHECK: vcvtudq2ph  -512(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x7a,0x72,0x80]
          vcvtudq2ph  -512(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vcvtuqq2ph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0xff,0x08,0x7a,0xf5]
          vcvtuqq2ph %xmm5, %xmm6

// CHECK: vcvtuqq2ph %ymm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0xff,0x28,0x7a,0xf5]
          vcvtuqq2ph %ymm5, %xmm6

// CHECK: vcvtuqq2phx  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0xff,0x0f,0x7a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtuqq2phx  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtuqq2ph  (%ecx){1to2}, %xmm6
// CHECK: encoding: [0x62,0xf5,0xff,0x18,0x7a,0x31]
          vcvtuqq2ph  (%ecx){1to2}, %xmm6

// CHECK: vcvtuqq2phx  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0xff,0x08,0x7a,0x71,0x7f]
          vcvtuqq2phx  2032(%ecx), %xmm6

// CHECK: vcvtuqq2ph  -1024(%edx){1to2}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xff,0x9f,0x7a,0x72,0x80]
          vcvtuqq2ph  -1024(%edx){1to2}, %xmm6 {%k7} {z}

// CHECK: vcvtuqq2ph  (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf5,0xff,0x38,0x7a,0x31]
          vcvtuqq2ph  (%ecx){1to4}, %xmm6

// CHECK: vcvtuqq2phy  4064(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0xff,0x28,0x7a,0x71,0x7f]
          vcvtuqq2phy  4064(%ecx), %xmm6

// CHECK: vcvtuqq2ph  -1024(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0xff,0xbf,0x7a,0x72,0x80]
          vcvtuqq2ph  -1024(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvtuw2ph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x7d,0xf5]
          vcvtuw2ph %xmm5, %xmm6

// CHECK: vcvtuw2ph %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x7d,0xf5]
          vcvtuw2ph %ymm5, %ymm6

// CHECK: vcvtuw2ph  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtuw2ph  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtuw2ph  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x7d,0x31]
          vcvtuw2ph  (%ecx){1to8}, %xmm6

// CHECK: vcvtuw2ph  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x7d,0x71,0x7f]
          vcvtuw2ph  2032(%ecx), %xmm6

// CHECK: vcvtuw2ph  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x7d,0x72,0x80]
          vcvtuw2ph  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vcvtuw2ph  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtuw2ph  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtuw2ph  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x7d,0x31]
          vcvtuw2ph  (%ecx){1to16}, %ymm6

// CHECK: vcvtuw2ph  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x7d,0x71,0x7f]
          vcvtuw2ph  4064(%ecx), %ymm6

// CHECK: vcvtuw2ph  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x7d,0x72,0x80]
          vcvtuw2ph  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vcvtw2ph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x7d,0xf5]
          vcvtw2ph %xmm5, %xmm6

// CHECK: vcvtw2ph %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x7d,0xf5]
          vcvtw2ph %ymm5, %ymm6

// CHECK: vcvtw2ph  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtw2ph  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtw2ph  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x7d,0x31]
          vcvtw2ph  (%ecx){1to8}, %xmm6

// CHECK: vcvtw2ph  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x7d,0x71,0x7f]
          vcvtw2ph  2032(%ecx), %xmm6

// CHECK: vcvtw2ph  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x9f,0x7d,0x72,0x80]
          vcvtw2ph  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vcvtw2ph  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x2f,0x7d,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtw2ph  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtw2ph  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x7d,0x31]
          vcvtw2ph  (%ecx){1to16}, %ymm6

// CHECK: vcvtw2ph  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x7d,0x71,0x7f]
          vcvtw2ph  4064(%ecx), %ymm6

// CHECK: vcvtw2ph  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xbf,0x7d,0x72,0x80]
          vcvtw2ph  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vfpclassph $123, %xmm6, %k5
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x66,0xee,0x7b]
          vfpclassph $123, %xmm6, %k5

// CHECK: vfpclassph $123, %ymm6, %k5
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x66,0xee,0x7b]
          vfpclassph $123, %ymm6, %k5

// CHECK: vfpclassphx  $123, 268435456(%esp,%esi,8), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x0f,0x66,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vfpclassphx  $123, 268435456(%esp,%esi,8), %k5 {%k7}

// CHECK: vfpclassph  $123, (%ecx){1to8}, %k5
// CHECK: encoding: [0x62,0xf3,0x7c,0x18,0x66,0x29,0x7b]
          vfpclassph  $123, (%ecx){1to8}, %k5

// CHECK: vfpclassphx  $123, 2032(%ecx), %k5
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x66,0x69,0x7f,0x7b]
          vfpclassphx  $123, 2032(%ecx), %k5

// CHECK: vfpclassph  $123, -256(%edx){1to8}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x1f,0x66,0x6a,0x80,0x7b]
          vfpclassph  $123, -256(%edx){1to8}, %k5 {%k7}

// CHECK: vfpclassph  $123, (%ecx){1to16}, %k5
// CHECK: encoding: [0x62,0xf3,0x7c,0x38,0x66,0x29,0x7b]
          vfpclassph  $123, (%ecx){1to16}, %k5

// CHECK: vfpclassphy  $123, 4064(%ecx), %k5
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x66,0x69,0x7f,0x7b]
          vfpclassphy  $123, 4064(%ecx), %k5

// CHECK: vfpclassph  $123, -256(%edx){1to16}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x3f,0x66,0x6a,0x80,0x7b]
          vfpclassph  $123, -256(%edx){1to16}, %k5 {%k7}

// CHECK: vgetexpph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x08,0x42,0xf5]
          vgetexpph %xmm5, %xmm6

// CHECK: vgetexpph %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x28,0x42,0xf5]
          vgetexpph %ymm5, %ymm6

// CHECK: vgetexpph  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7d,0x0f,0x42,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vgetexpph  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vgetexpph  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x18,0x42,0x31]
          vgetexpph  (%ecx){1to8}, %xmm6

// CHECK: vgetexpph  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x08,0x42,0x71,0x7f]
          vgetexpph  2032(%ecx), %xmm6

// CHECK: vgetexpph  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7d,0x9f,0x42,0x72,0x80]
          vgetexpph  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vgetexpph  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7d,0x2f,0x42,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vgetexpph  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vgetexpph  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x38,0x42,0x31]
          vgetexpph  (%ecx){1to16}, %ymm6

// CHECK: vgetexpph  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x28,0x42,0x71,0x7f]
          vgetexpph  4064(%ecx), %ymm6

// CHECK: vgetexpph  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7d,0xbf,0x42,0x72,0x80]
          vgetexpph  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vgetmantph $123, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x26,0xf5,0x7b]
          vgetmantph $123, %ymm5, %ymm6

// CHECK: vgetmantph $123, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x26,0xf5,0x7b]
          vgetmantph $123, %xmm5, %xmm6

// CHECK: vgetmantph  $123, 268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x0f,0x26,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantph  $123, 268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vgetmantph  $123, (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x18,0x26,0x31,0x7b]
          vgetmantph  $123, (%ecx){1to8}, %xmm6

// CHECK: vgetmantph  $123, 2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x26,0x71,0x7f,0x7b]
          vgetmantph  $123, 2032(%ecx), %xmm6

// CHECK: vgetmantph  $123, -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7c,0x9f,0x26,0x72,0x80,0x7b]
          vgetmantph  $123, -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vgetmantph  $123, 268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x2f,0x26,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantph  $123, 268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vgetmantph  $123, (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x38,0x26,0x31,0x7b]
          vgetmantph  $123, (%ecx){1to16}, %ymm6

// CHECK: vgetmantph  $123, 4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x26,0x71,0x7f,0x7b]
          vgetmantph  $123, 4064(%ecx), %ymm6

// CHECK: vgetmantph  $123, -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7c,0xbf,0x26,0x72,0x80,0x7b]
          vgetmantph  $123, -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vrcpph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x08,0x4c,0xf5]
          vrcpph %xmm5, %xmm6

// CHECK: vrcpph %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x28,0x4c,0xf5]
          vrcpph %ymm5, %ymm6

// CHECK: vrcpph  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7d,0x0f,0x4c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vrcpph  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vrcpph  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x18,0x4c,0x31]
          vrcpph  (%ecx){1to8}, %xmm6

// CHECK: vrcpph  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x08,0x4c,0x71,0x7f]
          vrcpph  2032(%ecx), %xmm6

// CHECK: vrcpph  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7d,0x9f,0x4c,0x72,0x80]
          vrcpph  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vrcpph  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7d,0x2f,0x4c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vrcpph  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vrcpph  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x38,0x4c,0x31]
          vrcpph  (%ecx){1to16}, %ymm6

// CHECK: vrcpph  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x28,0x4c,0x71,0x7f]
          vrcpph  4064(%ecx), %ymm6

// CHECK: vrcpph  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7d,0xbf,0x4c,0x72,0x80]
          vrcpph  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vreduceph $123, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x56,0xf5,0x7b]
          vreduceph $123, %ymm5, %ymm6

// CHECK: vreduceph $123, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x56,0xf5,0x7b]
          vreduceph $123, %xmm5, %xmm6

// CHECK: vreduceph  $123, 268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x0f,0x56,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreduceph  $123, 268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vreduceph  $123, (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x18,0x56,0x31,0x7b]
          vreduceph  $123, (%ecx){1to8}, %xmm6

// CHECK: vreduceph  $123, 2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x56,0x71,0x7f,0x7b]
          vreduceph  $123, 2032(%ecx), %xmm6

// CHECK: vreduceph  $123, -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7c,0x9f,0x56,0x72,0x80,0x7b]
          vreduceph  $123, -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vreduceph  $123, 268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x2f,0x56,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreduceph  $123, 268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vreduceph  $123, (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x38,0x56,0x31,0x7b]
          vreduceph  $123, (%ecx){1to16}, %ymm6

// CHECK: vreduceph  $123, 4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x56,0x71,0x7f,0x7b]
          vreduceph  $123, 4064(%ecx), %ymm6

// CHECK: vreduceph  $123, -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7c,0xbf,0x56,0x72,0x80,0x7b]
          vreduceph  $123, -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vrndscaleph $123, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x08,0xf5,0x7b]
          vrndscaleph $123, %ymm5, %ymm6

// CHECK: vrndscaleph $123, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x08,0xf5,0x7b]
          vrndscaleph $123, %xmm5, %xmm6

// CHECK: vrndscaleph  $123, 268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x0f,0x08,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscaleph  $123, 268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vrndscaleph  $123, (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x18,0x08,0x31,0x7b]
          vrndscaleph  $123, (%ecx){1to8}, %xmm6

// CHECK: vrndscaleph  $123, 2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x08,0x08,0x71,0x7f,0x7b]
          vrndscaleph  $123, 2032(%ecx), %xmm6

// CHECK: vrndscaleph  $123, -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7c,0x9f,0x08,0x72,0x80,0x7b]
          vrndscaleph  $123, -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vrndscaleph  $123, 268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7c,0x2f,0x08,0xb4,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscaleph  $123, 268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vrndscaleph  $123, (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x38,0x08,0x31,0x7b]
          vrndscaleph  $123, (%ecx){1to16}, %ymm6

// CHECK: vrndscaleph  $123, 4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf3,0x7c,0x28,0x08,0x71,0x7f,0x7b]
          vrndscaleph  $123, 4064(%ecx), %ymm6

// CHECK: vrndscaleph  $123, -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7c,0xbf,0x08,0x72,0x80,0x7b]
          vrndscaleph  $123, -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vrsqrtph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x08,0x4e,0xf5]
          vrsqrtph %xmm5, %xmm6

// CHECK: vrsqrtph %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x28,0x4e,0xf5]
          vrsqrtph %ymm5, %ymm6

// CHECK: vrsqrtph  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7d,0x0f,0x4e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtph  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vrsqrtph  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x18,0x4e,0x31]
          vrsqrtph  (%ecx){1to8}, %xmm6

// CHECK: vrsqrtph  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x08,0x4e,0x71,0x7f]
          vrsqrtph  2032(%ecx), %xmm6

// CHECK: vrsqrtph  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7d,0x9f,0x4e,0x72,0x80]
          vrsqrtph  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vrsqrtph  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7d,0x2f,0x4e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtph  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vrsqrtph  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x38,0x4e,0x31]
          vrsqrtph  (%ecx){1to16}, %ymm6

// CHECK: vrsqrtph  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf6,0x7d,0x28,0x4e,0x71,0x7f]
          vrsqrtph  4064(%ecx), %ymm6

// CHECK: vrsqrtph  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7d,0xbf,0x4e,0x72,0x80]
          vrsqrtph  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vscalefph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x2c,0xf4]
          vscalefph %ymm4, %ymm5, %ymm6

// CHECK: vscalefph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x2c,0xf4]
          vscalefph %xmm4, %xmm5, %xmm6

// CHECK: vscalefph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0x2c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vscalefph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vscalefph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0x2c,0x31]
          vscalefph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vscalefph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x2c,0x71,0x7f]
          vscalefph  4064(%ecx), %ymm5, %ymm6

// CHECK: vscalefph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0x2c,0x72,0x80]
          vscalefph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vscalefph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x2c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vscalefph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vscalefph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x2c,0x31]
          vscalefph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vscalefph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x2c,0x71,0x7f]
          vscalefph  2032(%ecx), %xmm5, %xmm6

// CHECK: vscalefph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0x2c,0x72,0x80]
          vscalefph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vsqrtph %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x51,0xf5]
          vsqrtph %xmm5, %xmm6

// CHECK: vsqrtph %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x51,0xf5]
          vsqrtph %ymm5, %ymm6

// CHECK: vsqrtph  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x51,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vsqrtph  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vsqrtph  (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x51,0x31]
          vsqrtph  (%ecx){1to8}, %xmm6

// CHECK: vsqrtph  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x51,0x71,0x7f]
          vsqrtph  2032(%ecx), %xmm6

// CHECK: vsqrtph  -256(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x51,0x72,0x80]
          vsqrtph  -256(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vsqrtph  268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x51,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vsqrtph  268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vsqrtph  (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x51,0x31]
          vsqrtph  (%ecx){1to16}, %ymm6

// CHECK: vsqrtph  4064(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x51,0x71,0x7f]
          vsqrtph  4064(%ecx), %ymm6

// CHECK: vsqrtph  -256(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x51,0x72,0x80]
          vsqrtph  -256(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vfmadd132ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x98,0xf4]
          vfmadd132ph %ymm4, %ymm5, %ymm6

// CHECK: vfmadd132ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x98,0xf4]
          vfmadd132ph %xmm4, %xmm5, %xmm6

// CHECK: vfmadd132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0x98,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmadd132ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0x98,0x31]
          vfmadd132ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmadd132ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x98,0x71,0x7f]
          vfmadd132ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmadd132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0x98,0x72,0x80]
          vfmadd132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmadd132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x98,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmadd132ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x98,0x31]
          vfmadd132ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmadd132ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x98,0x71,0x7f]
          vfmadd132ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmadd132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0x98,0x72,0x80]
          vfmadd132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmadd213ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xa8,0xf4]
          vfmadd213ph %ymm4, %ymm5, %ymm6

// CHECK: vfmadd213ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xa8,0xf4]
          vfmadd213ph %xmm4, %xmm5, %xmm6

// CHECK: vfmadd213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xa8,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmadd213ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xa8,0x31]
          vfmadd213ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmadd213ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xa8,0x71,0x7f]
          vfmadd213ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmadd213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xa8,0x72,0x80]
          vfmadd213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmadd213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xa8,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmadd213ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xa8,0x31]
          vfmadd213ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmadd213ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xa8,0x71,0x7f]
          vfmadd213ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmadd213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xa8,0x72,0x80]
          vfmadd213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmadd231ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xb8,0xf4]
          vfmadd231ph %ymm4, %ymm5, %ymm6

// CHECK: vfmadd231ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xb8,0xf4]
          vfmadd231ph %xmm4, %xmm5, %xmm6

// CHECK: vfmadd231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xb8,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmadd231ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xb8,0x31]
          vfmadd231ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmadd231ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xb8,0x71,0x7f]
          vfmadd231ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmadd231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xb8,0x72,0x80]
          vfmadd231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmadd231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xb8,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmadd231ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xb8,0x31]
          vfmadd231ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmadd231ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xb8,0x71,0x7f]
          vfmadd231ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmadd231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xb8,0x72,0x80]
          vfmadd231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmaddsub132ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x96,0xf4]
          vfmaddsub132ph %ymm4, %ymm5, %ymm6

// CHECK: vfmaddsub132ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x96,0xf4]
          vfmaddsub132ph %xmm4, %xmm5, %xmm6

// CHECK: vfmaddsub132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0x96,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddsub132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmaddsub132ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0x96,0x31]
          vfmaddsub132ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmaddsub132ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x96,0x71,0x7f]
          vfmaddsub132ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmaddsub132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0x96,0x72,0x80]
          vfmaddsub132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmaddsub132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x96,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddsub132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmaddsub132ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x96,0x31]
          vfmaddsub132ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmaddsub132ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x96,0x71,0x7f]
          vfmaddsub132ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmaddsub132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0x96,0x72,0x80]
          vfmaddsub132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmaddsub213ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xa6,0xf4]
          vfmaddsub213ph %ymm4, %ymm5, %ymm6

// CHECK: vfmaddsub213ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xa6,0xf4]
          vfmaddsub213ph %xmm4, %xmm5, %xmm6

// CHECK: vfmaddsub213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xa6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddsub213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmaddsub213ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xa6,0x31]
          vfmaddsub213ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmaddsub213ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xa6,0x71,0x7f]
          vfmaddsub213ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmaddsub213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xa6,0x72,0x80]
          vfmaddsub213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmaddsub213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xa6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddsub213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmaddsub213ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xa6,0x31]
          vfmaddsub213ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmaddsub213ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xa6,0x71,0x7f]
          vfmaddsub213ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmaddsub213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xa6,0x72,0x80]
          vfmaddsub213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmaddsub231ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xb6,0xf4]
          vfmaddsub231ph %ymm4, %ymm5, %ymm6

// CHECK: vfmaddsub231ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xb6,0xf4]
          vfmaddsub231ph %xmm4, %xmm5, %xmm6

// CHECK: vfmaddsub231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xb6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddsub231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmaddsub231ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xb6,0x31]
          vfmaddsub231ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmaddsub231ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xb6,0x71,0x7f]
          vfmaddsub231ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmaddsub231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xb6,0x72,0x80]
          vfmaddsub231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmaddsub231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xb6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddsub231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmaddsub231ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xb6,0x31]
          vfmaddsub231ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmaddsub231ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xb6,0x71,0x7f]
          vfmaddsub231ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmaddsub231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xb6,0x72,0x80]
          vfmaddsub231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmsub132ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x9a,0xf4]
          vfmsub132ph %ymm4, %ymm5, %ymm6

// CHECK: vfmsub132ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9a,0xf4]
          vfmsub132ph %xmm4, %xmm5, %xmm6

// CHECK: vfmsub132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0x9a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmsub132ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0x9a,0x31]
          vfmsub132ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmsub132ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x9a,0x71,0x7f]
          vfmsub132ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmsub132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0x9a,0x72,0x80]
          vfmsub132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmsub132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x9a,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmsub132ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x9a,0x31]
          vfmsub132ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmsub132ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9a,0x71,0x7f]
          vfmsub132ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmsub132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0x9a,0x72,0x80]
          vfmsub132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmsub213ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xaa,0xf4]
          vfmsub213ph %ymm4, %ymm5, %ymm6

// CHECK: vfmsub213ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xaa,0xf4]
          vfmsub213ph %xmm4, %xmm5, %xmm6

// CHECK: vfmsub213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xaa,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmsub213ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xaa,0x31]
          vfmsub213ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmsub213ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xaa,0x71,0x7f]
          vfmsub213ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmsub213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xaa,0x72,0x80]
          vfmsub213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmsub213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xaa,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmsub213ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xaa,0x31]
          vfmsub213ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmsub213ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xaa,0x71,0x7f]
          vfmsub213ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmsub213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xaa,0x72,0x80]
          vfmsub213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmsub231ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xba,0xf4]
          vfmsub231ph %ymm4, %ymm5, %ymm6

// CHECK: vfmsub231ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xba,0xf4]
          vfmsub231ph %xmm4, %xmm5, %xmm6

// CHECK: vfmsub231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xba,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmsub231ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xba,0x31]
          vfmsub231ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmsub231ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xba,0x71,0x7f]
          vfmsub231ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmsub231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xba,0x72,0x80]
          vfmsub231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmsub231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xba,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmsub231ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xba,0x31]
          vfmsub231ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmsub231ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xba,0x71,0x7f]
          vfmsub231ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmsub231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xba,0x72,0x80]
          vfmsub231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmsubadd132ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x97,0xf4]
          vfmsubadd132ph %ymm4, %ymm5, %ymm6

// CHECK: vfmsubadd132ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x97,0xf4]
          vfmsubadd132ph %xmm4, %xmm5, %xmm6

// CHECK: vfmsubadd132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0x97,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsubadd132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmsubadd132ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0x97,0x31]
          vfmsubadd132ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmsubadd132ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x97,0x71,0x7f]
          vfmsubadd132ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmsubadd132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0x97,0x72,0x80]
          vfmsubadd132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmsubadd132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x97,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsubadd132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmsubadd132ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x97,0x31]
          vfmsubadd132ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmsubadd132ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x97,0x71,0x7f]
          vfmsubadd132ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmsubadd132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0x97,0x72,0x80]
          vfmsubadd132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmsubadd213ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xa7,0xf4]
          vfmsubadd213ph %ymm4, %ymm5, %ymm6

// CHECK: vfmsubadd213ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xa7,0xf4]
          vfmsubadd213ph %xmm4, %xmm5, %xmm6

// CHECK: vfmsubadd213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xa7,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsubadd213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmsubadd213ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xa7,0x31]
          vfmsubadd213ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmsubadd213ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xa7,0x71,0x7f]
          vfmsubadd213ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmsubadd213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xa7,0x72,0x80]
          vfmsubadd213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmsubadd213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xa7,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsubadd213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmsubadd213ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xa7,0x31]
          vfmsubadd213ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmsubadd213ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xa7,0x71,0x7f]
          vfmsubadd213ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmsubadd213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xa7,0x72,0x80]
          vfmsubadd213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmsubadd231ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xb7,0xf4]
          vfmsubadd231ph %ymm4, %ymm5, %ymm6

// CHECK: vfmsubadd231ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xb7,0xf4]
          vfmsubadd231ph %xmm4, %xmm5, %xmm6

// CHECK: vfmsubadd231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xb7,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsubadd231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmsubadd231ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xb7,0x31]
          vfmsubadd231ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfmsubadd231ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xb7,0x71,0x7f]
          vfmsubadd231ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmsubadd231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xb7,0x72,0x80]
          vfmsubadd231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmsubadd231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xb7,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmsubadd231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmsubadd231ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xb7,0x31]
          vfmsubadd231ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfmsubadd231ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xb7,0x71,0x7f]
          vfmsubadd231ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmsubadd231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xb7,0x72,0x80]
          vfmsubadd231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfnmadd132ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x9c,0xf4]
          vfnmadd132ph %ymm4, %ymm5, %ymm6

// CHECK: vfnmadd132ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9c,0xf4]
          vfnmadd132ph %xmm4, %xmm5, %xmm6

// CHECK: vfnmadd132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0x9c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfnmadd132ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0x9c,0x31]
          vfnmadd132ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfnmadd132ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x9c,0x71,0x7f]
          vfnmadd132ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfnmadd132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0x9c,0x72,0x80]
          vfnmadd132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfnmadd132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x9c,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfnmadd132ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x9c,0x31]
          vfnmadd132ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfnmadd132ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9c,0x71,0x7f]
          vfnmadd132ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfnmadd132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0x9c,0x72,0x80]
          vfnmadd132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfnmadd213ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xac,0xf4]
          vfnmadd213ph %ymm4, %ymm5, %ymm6

// CHECK: vfnmadd213ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xac,0xf4]
          vfnmadd213ph %xmm4, %xmm5, %xmm6

// CHECK: vfnmadd213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xac,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfnmadd213ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xac,0x31]
          vfnmadd213ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfnmadd213ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xac,0x71,0x7f]
          vfnmadd213ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfnmadd213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xac,0x72,0x80]
          vfnmadd213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfnmadd213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xac,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfnmadd213ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xac,0x31]
          vfnmadd213ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfnmadd213ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xac,0x71,0x7f]
          vfnmadd213ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfnmadd213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xac,0x72,0x80]
          vfnmadd213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfnmadd231ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xbc,0xf4]
          vfnmadd231ph %ymm4, %ymm5, %ymm6

// CHECK: vfnmadd231ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbc,0xf4]
          vfnmadd231ph %xmm4, %xmm5, %xmm6

// CHECK: vfnmadd231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xbc,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfnmadd231ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xbc,0x31]
          vfnmadd231ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfnmadd231ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xbc,0x71,0x7f]
          vfnmadd231ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfnmadd231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xbc,0x72,0x80]
          vfnmadd231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfnmadd231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xbc,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfnmadd231ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xbc,0x31]
          vfnmadd231ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfnmadd231ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbc,0x71,0x7f]
          vfnmadd231ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfnmadd231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xbc,0x72,0x80]
          vfnmadd231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfnmsub132ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x9e,0xf4]
          vfnmsub132ph %ymm4, %ymm5, %ymm6

// CHECK: vfnmsub132ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9e,0xf4]
          vfnmsub132ph %xmm4, %xmm5, %xmm6

// CHECK: vfnmsub132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0x9e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfnmsub132ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0x9e,0x31]
          vfnmsub132ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfnmsub132ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0x9e,0x71,0x7f]
          vfnmsub132ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfnmsub132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0x9e,0x72,0x80]
          vfnmsub132ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfnmsub132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0x9e,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfnmsub132ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0x9e,0x31]
          vfnmsub132ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfnmsub132ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0x9e,0x71,0x7f]
          vfnmsub132ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfnmsub132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0x9e,0x72,0x80]
          vfnmsub132ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfnmsub213ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xae,0xf4]
          vfnmsub213ph %ymm4, %ymm5, %ymm6

// CHECK: vfnmsub213ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xae,0xf4]
          vfnmsub213ph %xmm4, %xmm5, %xmm6

// CHECK: vfnmsub213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xae,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfnmsub213ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xae,0x31]
          vfnmsub213ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfnmsub213ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xae,0x71,0x7f]
          vfnmsub213ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfnmsub213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xae,0x72,0x80]
          vfnmsub213ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfnmsub213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xae,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfnmsub213ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xae,0x31]
          vfnmsub213ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfnmsub213ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xae,0x71,0x7f]
          vfnmsub213ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfnmsub213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xae,0x72,0x80]
          vfnmsub213ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfnmsub231ph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xbe,0xf4]
          vfnmsub231ph %ymm4, %ymm5, %ymm6

// CHECK: vfnmsub231ph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbe,0xf4]
          vfnmsub231ph %xmm4, %xmm5, %xmm6

// CHECK: vfnmsub231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x2f,0xbe,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231ph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfnmsub231ph  (%ecx){1to16}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x38,0xbe,0x31]
          vfnmsub231ph  (%ecx){1to16}, %ymm5, %ymm6

// CHECK: vfnmsub231ph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x55,0x28,0xbe,0x71,0x7f]
          vfnmsub231ph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfnmsub231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0xbf,0xbe,0x72,0x80]
          vfnmsub231ph  -256(%edx){1to16}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfnmsub231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x55,0x0f,0xbe,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231ph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfnmsub231ph  (%ecx){1to8}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x18,0xbe,0x31]
          vfnmsub231ph  (%ecx){1to8}, %xmm5, %xmm6

// CHECK: vfnmsub231ph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x55,0x08,0xbe,0x71,0x7f]
          vfnmsub231ph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfnmsub231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x55,0x9f,0xbe,0x72,0x80]
          vfnmsub231ph  -256(%edx){1to8}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfcmaddcph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x57,0x28,0x56,0xf4]
          vfcmaddcph %ymm4, %ymm5, %ymm6

// CHECK: vfcmaddcph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0x56,0xf4]
          vfcmaddcph %xmm4, %xmm5, %xmm6

// CHECK: vfcmaddcph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x57,0x2f,0x56,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfcmaddcph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfcmaddcph  (%ecx){1to8}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x57,0x38,0x56,0x31]
          vfcmaddcph  (%ecx){1to8}, %ymm5, %ymm6

// CHECK: vfcmaddcph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x57,0x28,0x56,0x71,0x7f]
          vfcmaddcph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfcmaddcph  -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x57,0xbf,0x56,0x72,0x80]
          vfcmaddcph  -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfcmaddcph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x57,0x0f,0x56,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfcmaddcph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfcmaddcph  (%ecx){1to4}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x57,0x18,0x56,0x31]
          vfcmaddcph  (%ecx){1to4}, %xmm5, %xmm6

// CHECK: vfcmaddcph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0x56,0x71,0x7f]
          vfcmaddcph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfcmaddcph  -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x57,0x9f,0x56,0x72,0x80]
          vfcmaddcph  -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfcmulcph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x57,0x28,0xd6,0xf4]
          vfcmulcph %ymm4, %ymm5, %ymm6

// CHECK: vfcmulcph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0xd6,0xf4]
          vfcmulcph %xmm4, %xmm5, %xmm6

// CHECK: vfcmulcph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x57,0x2f,0xd6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfcmulcph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfcmulcph  (%ecx){1to8}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x57,0x38,0xd6,0x31]
          vfcmulcph  (%ecx){1to8}, %ymm5, %ymm6

// CHECK: vfcmulcph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x57,0x28,0xd6,0x71,0x7f]
          vfcmulcph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfcmulcph  -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x57,0xbf,0xd6,0x72,0x80]
          vfcmulcph  -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfcmulcph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x57,0x0f,0xd6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfcmulcph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfcmulcph  (%ecx){1to4}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x57,0x18,0xd6,0x31]
          vfcmulcph  (%ecx){1to4}, %xmm5, %xmm6

// CHECK: vfcmulcph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x57,0x08,0xd6,0x71,0x7f]
          vfcmulcph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfcmulcph  -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x57,0x9f,0xd6,0x72,0x80]
          vfcmulcph  -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmaddcph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x56,0x28,0x56,0xf4]
          vfmaddcph %ymm4, %ymm5, %ymm6

// CHECK: vfmaddcph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0x56,0xf4]
          vfmaddcph %xmm4, %xmm5, %xmm6

// CHECK: vfmaddcph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x56,0x2f,0x56,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddcph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmaddcph  (%ecx){1to8}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x56,0x38,0x56,0x31]
          vfmaddcph  (%ecx){1to8}, %ymm5, %ymm6

// CHECK: vfmaddcph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x56,0x28,0x56,0x71,0x7f]
          vfmaddcph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmaddcph  -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x56,0xbf,0x56,0x72,0x80]
          vfmaddcph  -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmaddcph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x56,0x0f,0x56,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmaddcph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmaddcph  (%ecx){1to4}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x56,0x18,0x56,0x31]
          vfmaddcph  (%ecx){1to4}, %xmm5, %xmm6

// CHECK: vfmaddcph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0x56,0x71,0x7f]
          vfmaddcph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmaddcph  -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x56,0x9f,0x56,0x72,0x80]
          vfmaddcph  -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}

// CHECK: vfmulcph %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x56,0x28,0xd6,0xf4]
          vfmulcph %ymm4, %ymm5, %ymm6

// CHECK: vfmulcph %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0xd6,0xf4]
          vfmulcph %xmm4, %xmm5, %xmm6

// CHECK: vfmulcph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x56,0x2f,0xd6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmulcph  268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vfmulcph  (%ecx){1to8}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x56,0x38,0xd6,0x31]
          vfmulcph  (%ecx){1to8}, %ymm5, %ymm6

// CHECK: vfmulcph  4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf6,0x56,0x28,0xd6,0x71,0x7f]
          vfmulcph  4064(%ecx), %ymm5, %ymm6

// CHECK: vfmulcph  -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x56,0xbf,0xd6,0x72,0x80]
          vfmulcph  -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vfmulcph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf6,0x56,0x0f,0xd6,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vfmulcph  268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vfmulcph  (%ecx){1to4}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x56,0x18,0xd6,0x31]
          vfmulcph  (%ecx){1to4}, %xmm5, %xmm6

// CHECK: vfmulcph  2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf6,0x56,0x08,0xd6,0x71,0x7f]
          vfmulcph  2032(%ecx), %xmm5, %xmm6

// CHECK: vfmulcph  -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x56,0x9f,0xd6,0x72,0x80]
          vfmulcph  -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}
