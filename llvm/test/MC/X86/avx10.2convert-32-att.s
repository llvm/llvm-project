// RUN: llvm-mc -triple i386 --show-encoding %s | FileCheck %s

// CHECK: vcvt2ps2phx %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x65,0x28,0x67,0xd4]
          vcvt2ps2phx %ymm4, %ymm3, %ymm2

// CHECK: vcvt2ps2phx {rn-sae}, %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x61,0x18,0x67,0xd4]
          vcvt2ps2phx {rn-sae}, %ymm4, %ymm3, %ymm2

// CHECK: vcvt2ps2phx %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x65,0x2f,0x67,0xd4]
          vcvt2ps2phx %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ps2phx {rz-sae}, %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x61,0xff,0x67,0xd4]
          vcvt2ps2phx {rz-sae}, %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ps2phx %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0x67,0xd4]
          vcvt2ps2phx %zmm4, %zmm3, %zmm2

// CHECK: vcvt2ps2phx {rn-sae}, %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x65,0x18,0x67,0xd4]
          vcvt2ps2phx {rn-sae}, %zmm4, %zmm3, %zmm2

// CHECK: vcvt2ps2phx %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x65,0x4f,0x67,0xd4]
          vcvt2ps2phx %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ps2phx {rz-sae}, %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x65,0xff,0x67,0xd4]
          vcvt2ps2phx {rz-sae}, %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ps2phx %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x65,0x08,0x67,0xd4]
          vcvt2ps2phx %xmm4, %xmm3, %xmm2

// CHECK: vcvt2ps2phx %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x65,0x0f,0x67,0xd4]
          vcvt2ps2phx %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ps2phx %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x65,0x8f,0x67,0xd4]
          vcvt2ps2phx %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ps2phx  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0x67,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ps2phx  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vcvt2ps2phx  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x65,0x4f,0x67,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ps2phx  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ps2phx  (%eax){1to16}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x65,0x58,0x67,0x10]
          vcvt2ps2phx  (%eax){1to16}, %zmm3, %zmm2

// CHECK: vcvt2ps2phx  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x65,0x48,0x67,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ps2phx  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vcvt2ps2phx  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x65,0xcf,0x67,0x51,0x7f]
          vcvt2ps2phx  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ps2phx  -512(%edx){1to16}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x65,0xdf,0x67,0x52,0x80]
          vcvt2ps2phx  -512(%edx){1to16}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ps2phx  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x65,0x28,0x67,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ps2phx  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vcvt2ps2phx  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x65,0x2f,0x67,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ps2phx  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ps2phx  (%eax){1to8}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x65,0x38,0x67,0x10]
          vcvt2ps2phx  (%eax){1to8}, %ymm3, %ymm2

// CHECK: vcvt2ps2phx  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x65,0x28,0x67,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ps2phx  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vcvt2ps2phx  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x65,0xaf,0x67,0x51,0x7f]
          vcvt2ps2phx  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ps2phx  -512(%edx){1to8}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x65,0xbf,0x67,0x52,0x80]
          vcvt2ps2phx  -512(%edx){1to8}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ps2phx  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x65,0x08,0x67,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ps2phx  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vcvt2ps2phx  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x65,0x0f,0x67,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ps2phx  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ps2phx  (%eax){1to4}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x65,0x18,0x67,0x10]
          vcvt2ps2phx  (%eax){1to4}, %xmm3, %xmm2

// CHECK: vcvt2ps2phx  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x65,0x08,0x67,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ps2phx  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vcvt2ps2phx  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x65,0x8f,0x67,0x51,0x7f]
          vcvt2ps2phx  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ps2phx  -512(%edx){1to4}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x65,0x9f,0x67,0x52,0x80]
          vcvt2ps2phx  -512(%edx){1to4}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8 %zmm4, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x74,0xd4]
          vcvtbiasph2bf8 %zmm4, %zmm3, %ymm2

// CHECK: vcvtbiasph2bf8 %zmm4, %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x74,0xd4]
          vcvtbiasph2bf8 %zmm4, %zmm3, %ymm2 {%k7}

// CHECK: vcvtbiasph2bf8 %zmm4, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x74,0xd4]
          vcvtbiasph2bf8 %zmm4, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x64,0x08,0x74,0xd4]
          vcvtbiasph2bf8 %xmm4, %xmm3, %xmm2

// CHECK: vcvtbiasph2bf8 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x74,0xd4]
          vcvtbiasph2bf8 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2bf8 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x74,0xd4]
          vcvtbiasph2bf8 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8 %ymm4, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x64,0x28,0x74,0xd4]
          vcvtbiasph2bf8 %ymm4, %ymm3, %xmm2

// CHECK: vcvtbiasph2bf8 %ymm4, %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x74,0xd4]
          vcvtbiasph2bf8 %ymm4, %ymm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2bf8 %ymm4, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x74,0xd4]
          vcvtbiasph2bf8 %ymm4, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8  268435456(%esp,%esi,8), %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x64,0x28,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8  268435456(%esp,%esi,8), %ymm3, %xmm2

// CHECK: vcvtbiasph2bf8  291(%edi,%eax,4), %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x64,0x2f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8  291(%edi,%eax,4), %ymm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2bf8  (%eax){1to16}, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x64,0x38,0x74,0x10]
          vcvtbiasph2bf8  (%eax){1to16}, %ymm3, %xmm2

// CHECK: vcvtbiasph2bf8  -1024(,%ebp,2), %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x64,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2bf8  -1024(,%ebp,2), %ymm3, %xmm2

// CHECK: vcvtbiasph2bf8  4064(%ecx), %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x64,0xaf,0x74,0x51,0x7f]
          vcvtbiasph2bf8  4064(%ecx), %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8  -256(%edx){1to16}, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x64,0xbf,0x74,0x52,0x80]
          vcvtbiasph2bf8  -256(%edx){1to16}, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8  268435456(%esp,%esi,8), %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8  268435456(%esp,%esi,8), %zmm3, %ymm2

// CHECK: vcvtbiasph2bf8  291(%edi,%eax,4), %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x64,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8  291(%edi,%eax,4), %zmm3, %ymm2 {%k7}

// CHECK: vcvtbiasph2bf8  (%eax){1to32}, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x64,0x58,0x74,0x10]
          vcvtbiasph2bf8  (%eax){1to32}, %zmm3, %ymm2

// CHECK: vcvtbiasph2bf8  -2048(,%ebp,2), %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x64,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2bf8  -2048(,%ebp,2), %zmm3, %ymm2

// CHECK: vcvtbiasph2bf8  8128(%ecx), %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x64,0xcf,0x74,0x51,0x7f]
          vcvtbiasph2bf8  8128(%ecx), %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8  -256(%edx){1to32}, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x64,0xdf,0x74,0x52,0x80]
          vcvtbiasph2bf8  -256(%edx){1to32}, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x64,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vcvtbiasph2bf8  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x64,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2bf8  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x64,0x18,0x74,0x10]
          vcvtbiasph2bf8  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vcvtbiasph2bf8  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x64,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2bf8  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vcvtbiasph2bf8  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x64,0x8f,0x74,0x51,0x7f]
          vcvtbiasph2bf8  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x64,0x9f,0x74,0x52,0x80]
          vcvtbiasph2bf8  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8s %zmm4, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x74,0xd4]
          vcvtbiasph2bf8s %zmm4, %zmm3, %ymm2

// CHECK: vcvtbiasph2bf8s %zmm4, %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x74,0xd4]
          vcvtbiasph2bf8s %zmm4, %zmm3, %ymm2 {%k7}

// CHECK: vcvtbiasph2bf8s %zmm4, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x74,0xd4]
          vcvtbiasph2bf8s %zmm4, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8s %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x74,0xd4]
          vcvtbiasph2bf8s %xmm4, %xmm3, %xmm2

// CHECK: vcvtbiasph2bf8s %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x74,0xd4]
          vcvtbiasph2bf8s %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2bf8s %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x74,0xd4]
          vcvtbiasph2bf8s %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8s %ymm4, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x74,0xd4]
          vcvtbiasph2bf8s %ymm4, %ymm3, %xmm2

// CHECK: vcvtbiasph2bf8s %ymm4, %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x74,0xd4]
          vcvtbiasph2bf8s %ymm4, %ymm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2bf8s %ymm4, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x74,0xd4]
          vcvtbiasph2bf8s %ymm4, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  268435456(%esp,%esi,8), %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s  268435456(%esp,%esi,8), %ymm3, %xmm2

// CHECK: vcvtbiasph2bf8s  291(%edi,%eax,4), %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s  291(%edi,%eax,4), %ymm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2bf8s  (%eax){1to16}, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x38,0x74,0x10]
          vcvtbiasph2bf8s  (%eax){1to16}, %ymm3, %xmm2

// CHECK: vcvtbiasph2bf8s  -1024(,%ebp,2), %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2bf8s  -1024(,%ebp,2), %ymm3, %xmm2

// CHECK: vcvtbiasph2bf8s  4064(%ecx), %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x74,0x51,0x7f]
          vcvtbiasph2bf8s  4064(%ecx), %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  -256(%edx){1to16}, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xbf,0x74,0x52,0x80]
          vcvtbiasph2bf8s  -256(%edx){1to16}, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  268435456(%esp,%esi,8), %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s  268435456(%esp,%esi,8), %zmm3, %ymm2

// CHECK: vcvtbiasph2bf8s  291(%edi,%eax,4), %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s  291(%edi,%eax,4), %zmm3, %ymm2 {%k7}

// CHECK: vcvtbiasph2bf8s  (%eax){1to32}, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x58,0x74,0x10]
          vcvtbiasph2bf8s  (%eax){1to32}, %zmm3, %ymm2

// CHECK: vcvtbiasph2bf8s  -2048(,%ebp,2), %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2bf8s  -2048(,%ebp,2), %zmm3, %ymm2

// CHECK: vcvtbiasph2bf8s  8128(%ecx), %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x74,0x51,0x7f]
          vcvtbiasph2bf8s  8128(%ecx), %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  -256(%edx){1to32}, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xdf,0x74,0x52,0x80]
          vcvtbiasph2bf8s  -256(%edx){1to32}, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2bf8s  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vcvtbiasph2bf8s  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2bf8s  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2bf8s  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x18,0x74,0x10]
          vcvtbiasph2bf8s  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vcvtbiasph2bf8s  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2bf8s  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vcvtbiasph2bf8s  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x74,0x51,0x7f]
          vcvtbiasph2bf8s  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2bf8s  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0x9f,0x74,0x52,0x80]
          vcvtbiasph2bf8s  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8 %zmm4, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x18,0xd4]
          vcvtbiasph2hf8 %zmm4, %zmm3, %ymm2

// CHECK: vcvtbiasph2hf8 %zmm4, %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x18,0xd4]
          vcvtbiasph2hf8 %zmm4, %zmm3, %ymm2 {%k7}

// CHECK: vcvtbiasph2hf8 %zmm4, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x18,0xd4]
          vcvtbiasph2hf8 %zmm4, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x18,0xd4]
          vcvtbiasph2hf8 %xmm4, %xmm3, %xmm2

// CHECK: vcvtbiasph2hf8 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x18,0xd4]
          vcvtbiasph2hf8 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2hf8 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x18,0xd4]
          vcvtbiasph2hf8 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8 %ymm4, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x18,0xd4]
          vcvtbiasph2hf8 %ymm4, %ymm3, %xmm2

// CHECK: vcvtbiasph2hf8 %ymm4, %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x18,0xd4]
          vcvtbiasph2hf8 %ymm4, %ymm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2hf8 %ymm4, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x18,0xd4]
          vcvtbiasph2hf8 %ymm4, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8  268435456(%esp,%esi,8), %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8  268435456(%esp,%esi,8), %ymm3, %xmm2

// CHECK: vcvtbiasph2hf8  291(%edi,%eax,4), %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8  291(%edi,%eax,4), %ymm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2hf8  (%eax){1to16}, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x38,0x18,0x10]
          vcvtbiasph2hf8  (%eax){1to16}, %ymm3, %xmm2

// CHECK: vcvtbiasph2hf8  -1024(,%ebp,2), %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x18,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2hf8  -1024(,%ebp,2), %ymm3, %xmm2

// CHECK: vcvtbiasph2hf8  4064(%ecx), %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x18,0x51,0x7f]
          vcvtbiasph2hf8  4064(%ecx), %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8  -256(%edx){1to16}, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xbf,0x18,0x52,0x80]
          vcvtbiasph2hf8  -256(%edx){1to16}, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8  268435456(%esp,%esi,8), %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8  268435456(%esp,%esi,8), %zmm3, %ymm2

// CHECK: vcvtbiasph2hf8  291(%edi,%eax,4), %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8  291(%edi,%eax,4), %zmm3, %ymm2 {%k7}

// CHECK: vcvtbiasph2hf8  (%eax){1to32}, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x58,0x18,0x10]
          vcvtbiasph2hf8  (%eax){1to32}, %zmm3, %ymm2

// CHECK: vcvtbiasph2hf8  -2048(,%ebp,2), %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x18,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2hf8  -2048(,%ebp,2), %zmm3, %ymm2

// CHECK: vcvtbiasph2hf8  8128(%ecx), %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x18,0x51,0x7f]
          vcvtbiasph2hf8  8128(%ecx), %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8  -256(%edx){1to32}, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xdf,0x18,0x52,0x80]
          vcvtbiasph2hf8  -256(%edx){1to32}, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vcvtbiasph2hf8  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2hf8  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x18,0x18,0x10]
          vcvtbiasph2hf8  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vcvtbiasph2hf8  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x18,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2hf8  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vcvtbiasph2hf8  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x18,0x51,0x7f]
          vcvtbiasph2hf8  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0x9f,0x18,0x52,0x80]
          vcvtbiasph2hf8  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8s %zmm4, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x1b,0xd4]
          vcvtbiasph2hf8s %zmm4, %zmm3, %ymm2

// CHECK: vcvtbiasph2hf8s %zmm4, %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x1b,0xd4]
          vcvtbiasph2hf8s %zmm4, %zmm3, %ymm2 {%k7}

// CHECK: vcvtbiasph2hf8s %zmm4, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x1b,0xd4]
          vcvtbiasph2hf8s %zmm4, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8s %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x1b,0xd4]
          vcvtbiasph2hf8s %xmm4, %xmm3, %xmm2

// CHECK: vcvtbiasph2hf8s %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x1b,0xd4]
          vcvtbiasph2hf8s %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2hf8s %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x1b,0xd4]
          vcvtbiasph2hf8s %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8s %ymm4, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x1b,0xd4]
          vcvtbiasph2hf8s %ymm4, %ymm3, %xmm2

// CHECK: vcvtbiasph2hf8s %ymm4, %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x1b,0xd4]
          vcvtbiasph2hf8s %ymm4, %ymm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2hf8s %ymm4, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x1b,0xd4]
          vcvtbiasph2hf8s %ymm4, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  268435456(%esp,%esi,8), %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s  268435456(%esp,%esi,8), %ymm3, %xmm2

// CHECK: vcvtbiasph2hf8s  291(%edi,%eax,4), %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x2f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s  291(%edi,%eax,4), %ymm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2hf8s  (%eax){1to16}, %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x38,0x1b,0x10]
          vcvtbiasph2hf8s  (%eax){1to16}, %ymm3, %xmm2

// CHECK: vcvtbiasph2hf8s  -1024(,%ebp,2), %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x28,0x1b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbiasph2hf8s  -1024(,%ebp,2), %ymm3, %xmm2

// CHECK: vcvtbiasph2hf8s  4064(%ecx), %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xaf,0x1b,0x51,0x7f]
          vcvtbiasph2hf8s  4064(%ecx), %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  -256(%edx){1to16}, %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xbf,0x1b,0x52,0x80]
          vcvtbiasph2hf8s  -256(%edx){1to16}, %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  268435456(%esp,%esi,8), %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s  268435456(%esp,%esi,8), %zmm3, %ymm2

// CHECK: vcvtbiasph2hf8s  291(%edi,%eax,4), %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x4f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s  291(%edi,%eax,4), %zmm3, %ymm2 {%k7}

// CHECK: vcvtbiasph2hf8s  (%eax){1to32}, %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x58,0x1b,0x10]
          vcvtbiasph2hf8s  (%eax){1to32}, %zmm3, %ymm2

// CHECK: vcvtbiasph2hf8s  -2048(,%ebp,2), %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x64,0x48,0x1b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbiasph2hf8s  -2048(,%ebp,2), %zmm3, %ymm2

// CHECK: vcvtbiasph2hf8s  8128(%ecx), %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xcf,0x1b,0x51,0x7f]
          vcvtbiasph2hf8s  8128(%ecx), %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  -256(%edx){1to32}, %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0xdf,0x1b,0x52,0x80]
          vcvtbiasph2hf8s  -256(%edx){1to32}, %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbiasph2hf8s  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vcvtbiasph2hf8s  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x64,0x0f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbiasph2hf8s  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vcvtbiasph2hf8s  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x18,0x1b,0x10]
          vcvtbiasph2hf8s  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vcvtbiasph2hf8s  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x64,0x08,0x1b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbiasph2hf8s  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vcvtbiasph2hf8s  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0x8f,0x1b,0x51,0x7f]
          vcvtbiasph2hf8s  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbiasph2hf8s  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x64,0x9f,0x1b,0x52,0x80]
          vcvtbiasph2hf8s  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvthf82ph %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x1e,0xd3]
          vcvthf82ph %xmm3, %xmm2

// CHECK: vcvthf82ph %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x1e,0xd3]
          vcvthf82ph %xmm3, %xmm2 {%k7}

// CHECK: vcvthf82ph %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x1e,0xd3]
          vcvthf82ph %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvthf82ph %xmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x1e,0xd3]
          vcvthf82ph %xmm3, %ymm2

// CHECK: vcvthf82ph %xmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x1e,0xd3]
          vcvthf82ph %xmm3, %ymm2 {%k7}

// CHECK: vcvthf82ph %xmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x1e,0xd3]
          vcvthf82ph %xmm3, %ymm2 {%k7} {z}

// CHECK: vcvthf82ph %ymm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x1e,0xd3]
          vcvthf82ph %ymm3, %zmm2

// CHECK: vcvthf82ph %ymm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x1e,0xd3]
          vcvthf82ph %ymm3, %zmm2 {%k7}

// CHECK: vcvthf82ph %ymm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x1e,0xd3]
          vcvthf82ph %ymm3, %zmm2 {%k7} {z}

// CHECK: vcvthf82ph  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x1e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvthf82ph  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvthf82ph  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x1e,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvthf82ph  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvthf82ph  (%eax), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x1e,0x10]
          vcvthf82ph  (%eax), %xmm2

// CHECK: vcvthf82ph  -256(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x1e,0x14,0x6d,0x00,0xff,0xff,0xff]
          vcvthf82ph  -256(,%ebp,2), %xmm2

// CHECK: vcvthf82ph  1016(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x1e,0x51,0x7f]
          vcvthf82ph  1016(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvthf82ph  -1024(%edx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x1e,0x52,0x80]
          vcvthf82ph  -1024(%edx), %xmm2 {%k7} {z}

// CHECK: vcvthf82ph  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x1e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvthf82ph  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvthf82ph  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x1e,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvthf82ph  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvthf82ph  (%eax), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x1e,0x10]
          vcvthf82ph  (%eax), %ymm2

// CHECK: vcvthf82ph  -512(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x1e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvthf82ph  -512(,%ebp,2), %ymm2

// CHECK: vcvthf82ph  2032(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x1e,0x51,0x7f]
          vcvthf82ph  2032(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvthf82ph  -2048(%edx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x1e,0x52,0x80]
          vcvthf82ph  -2048(%edx), %ymm2 {%k7} {z}

// CHECK: vcvthf82ph  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x1e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvthf82ph  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvthf82ph  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x1e,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvthf82ph  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvthf82ph  (%eax), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x1e,0x10]
          vcvthf82ph  (%eax), %zmm2

// CHECK: vcvthf82ph  -1024(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x1e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvthf82ph  -1024(,%ebp,2), %zmm2

// CHECK: vcvthf82ph  4064(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x1e,0x51,0x7f]
          vcvthf82ph  4064(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvthf82ph  -4096(%edx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x1e,0x52,0x80]
          vcvthf82ph  -4096(%edx), %zmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x74,0xd4]
          vcvt2ph2bf8 %ymm4, %ymm3, %ymm2

// CHECK: vcvt2ph2bf8 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x67,0x2f,0x74,0xd4]
          vcvt2ph2bf8 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ph2bf8 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x67,0xaf,0x74,0xd4]
          vcvt2ph2bf8 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2bf8 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x74,0xd4]
          vcvt2ph2bf8 %zmm4, %zmm3, %zmm2

// CHECK: vcvt2ph2bf8 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x67,0x4f,0x74,0xd4]
          vcvt2ph2bf8 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ph2bf8 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x67,0xcf,0x74,0xd4]
          vcvt2ph2bf8 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x74,0xd4]
          vcvt2ph2bf8 %xmm4, %xmm3, %xmm2

// CHECK: vcvt2ph2bf8 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x67,0x0f,0x74,0xd4]
          vcvt2ph2bf8 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ph2bf8 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x67,0x8f,0x74,0xd4]
          vcvt2ph2bf8 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vcvt2ph2bf8  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x67,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ph2bf8  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x58,0x74,0x10]
          vcvt2ph2bf8  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vcvt2ph2bf8  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2bf8  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vcvt2ph2bf8  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x67,0xcf,0x74,0x51,0x7f]
          vcvt2ph2bf8  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x67,0xdf,0x74,0x52,0x80]
          vcvt2ph2bf8  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vcvt2ph2bf8  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x67,0x2f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ph2bf8  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x38,0x74,0x10]
          vcvt2ph2bf8  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vcvt2ph2bf8  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2bf8  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vcvt2ph2bf8  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x67,0xaf,0x74,0x51,0x7f]
          vcvt2ph2bf8  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2bf8  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x67,0xbf,0x74,0x52,0x80]
          vcvt2ph2bf8  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2bf8  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vcvt2ph2bf8  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x67,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ph2bf8  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x18,0x74,0x10]
          vcvt2ph2bf8  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vcvt2ph2bf8  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2bf8  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vcvt2ph2bf8  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x67,0x8f,0x74,0x51,0x7f]
          vcvt2ph2bf8  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x67,0x9f,0x74,0x52,0x80]
          vcvt2ph2bf8  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8s %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x74,0xd4]
          vcvt2ph2bf8s %ymm4, %ymm3, %ymm2

// CHECK: vcvt2ph2bf8s %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x74,0xd4]
          vcvt2ph2bf8s %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ph2bf8s %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x74,0xd4]
          vcvt2ph2bf8s %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2bf8s %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x74,0xd4]
          vcvt2ph2bf8s %zmm4, %zmm3, %zmm2

// CHECK: vcvt2ph2bf8s %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x74,0xd4]
          vcvt2ph2bf8s %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ph2bf8s %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x74,0xd4]
          vcvt2ph2bf8s %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8s %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x74,0xd4]
          vcvt2ph2bf8s %xmm4, %xmm3, %xmm2

// CHECK: vcvt2ph2bf8s %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x74,0xd4]
          vcvt2ph2bf8s %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ph2bf8s %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x74,0xd4]
          vcvt2ph2bf8s %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8s  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8s  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vcvt2ph2bf8s  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8s  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ph2bf8s  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x58,0x74,0x10]
          vcvt2ph2bf8s  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vcvt2ph2bf8s  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2bf8s  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vcvt2ph2bf8s  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x74,0x51,0x7f]
          vcvt2ph2bf8s  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8s  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xdf,0x74,0x52,0x80]
          vcvt2ph2bf8s  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8s  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8s  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vcvt2ph2bf8s  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8s  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ph2bf8s  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x38,0x74,0x10]
          vcvt2ph2bf8s  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vcvt2ph2bf8s  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2bf8s  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vcvt2ph2bf8s  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x74,0x51,0x7f]
          vcvt2ph2bf8s  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2bf8s  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xbf,0x74,0x52,0x80]
          vcvt2ph2bf8s  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2bf8s  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2bf8s  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vcvt2ph2bf8s  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2bf8s  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ph2bf8s  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x18,0x74,0x10]
          vcvt2ph2bf8s  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vcvt2ph2bf8s  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2bf8s  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vcvt2ph2bf8s  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x74,0x51,0x7f]
          vcvt2ph2bf8s  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2bf8s  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0x9f,0x74,0x52,0x80]
          vcvt2ph2bf8s  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x18,0xd4]
          vcvt2ph2hf8 %ymm4, %ymm3, %ymm2

// CHECK: vcvt2ph2hf8 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x18,0xd4]
          vcvt2ph2hf8 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ph2hf8 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x18,0xd4]
          vcvt2ph2hf8 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2hf8 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x18,0xd4]
          vcvt2ph2hf8 %zmm4, %zmm3, %zmm2

// CHECK: vcvt2ph2hf8 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x18,0xd4]
          vcvt2ph2hf8 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ph2hf8 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x18,0xd4]
          vcvt2ph2hf8 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x18,0xd4]
          vcvt2ph2hf8 %xmm4, %xmm3, %xmm2

// CHECK: vcvt2ph2hf8 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x18,0xd4]
          vcvt2ph2hf8 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ph2hf8 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x18,0xd4]
          vcvt2ph2hf8 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vcvt2ph2hf8  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ph2hf8  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x58,0x18,0x10]
          vcvt2ph2hf8  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vcvt2ph2hf8  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x18,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2hf8  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vcvt2ph2hf8  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x18,0x51,0x7f]
          vcvt2ph2hf8  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xdf,0x18,0x52,0x80]
          vcvt2ph2hf8  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vcvt2ph2hf8  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ph2hf8  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x38,0x18,0x10]
          vcvt2ph2hf8  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vcvt2ph2hf8  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x18,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2hf8  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vcvt2ph2hf8  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x18,0x51,0x7f]
          vcvt2ph2hf8  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2hf8  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xbf,0x18,0x52,0x80]
          vcvt2ph2hf8  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2hf8  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vcvt2ph2hf8  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ph2hf8  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x18,0x18,0x10]
          vcvt2ph2hf8  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vcvt2ph2hf8  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x18,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2hf8  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vcvt2ph2hf8  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x18,0x51,0x7f]
          vcvt2ph2hf8  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0x9f,0x18,0x52,0x80]
          vcvt2ph2hf8  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8s %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x1b,0xd4]
          vcvt2ph2hf8s %ymm4, %ymm3, %ymm2

// CHECK: vcvt2ph2hf8s %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x1b,0xd4]
          vcvt2ph2hf8s %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ph2hf8s %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x1b,0xd4]
          vcvt2ph2hf8s %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2hf8s %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x1b,0xd4]
          vcvt2ph2hf8s %zmm4, %zmm3, %zmm2

// CHECK: vcvt2ph2hf8s %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x1b,0xd4]
          vcvt2ph2hf8s %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ph2hf8s %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x1b,0xd4]
          vcvt2ph2hf8s %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8s %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x1b,0xd4]
          vcvt2ph2hf8s %xmm4, %xmm3, %xmm2

// CHECK: vcvt2ph2hf8s %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x1b,0xd4]
          vcvt2ph2hf8s %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ph2hf8s %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x1b,0xd4]
          vcvt2ph2hf8s %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8s  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8s  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vcvt2ph2hf8s  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x4f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8s  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vcvt2ph2hf8s  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x58,0x1b,0x10]
          vcvt2ph2hf8s  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vcvt2ph2hf8s  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x48,0x1b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvt2ph2hf8s  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vcvt2ph2hf8s  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xcf,0x1b,0x51,0x7f]
          vcvt2ph2hf8s  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8s  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xdf,0x1b,0x52,0x80]
          vcvt2ph2hf8s  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8s  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8s  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vcvt2ph2hf8s  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x2f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8s  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vcvt2ph2hf8s  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x38,0x1b,0x10]
          vcvt2ph2hf8s  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vcvt2ph2hf8s  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x67,0x28,0x1b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvt2ph2hf8s  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vcvt2ph2hf8s  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xaf,0x1b,0x51,0x7f]
          vcvt2ph2hf8s  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2hf8s  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0xbf,0x1b,0x52,0x80]
          vcvt2ph2hf8s  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvt2ph2hf8s  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvt2ph2hf8s  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vcvt2ph2hf8s  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x67,0x0f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvt2ph2hf8s  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vcvt2ph2hf8s  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x18,0x1b,0x10]
          vcvt2ph2hf8s  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vcvt2ph2hf8s  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x67,0x08,0x1b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvt2ph2hf8s  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vcvt2ph2hf8s  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0x8f,0x1b,0x51,0x7f]
          vcvt2ph2hf8s  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvt2ph2hf8s  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x67,0x9f,0x1b,0x52,0x80]
          vcvt2ph2hf8s  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8 %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x74,0xd3]
          vcvtph2bf8 %xmm3, %xmm2

// CHECK: vcvtph2bf8 %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x7e,0x0f,0x74,0xd3]
          vcvtph2bf8 %xmm3, %xmm2 {%k7}

// CHECK: vcvtph2bf8 %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0x8f,0x74,0xd3]
          vcvtph2bf8 %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8 %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x74,0xd3]
          vcvtph2bf8 %zmm3, %ymm2

// CHECK: vcvtph2bf8 %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x7e,0x4f,0x74,0xd3]
          vcvtph2bf8 %zmm3, %ymm2 {%k7}

// CHECK: vcvtph2bf8 %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0xcf,0x74,0xd3]
          vcvtph2bf8 %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtph2bf8 %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x74,0xd3]
          vcvtph2bf8 %ymm3, %xmm2

// CHECK: vcvtph2bf8 %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x7e,0x2f,0x74,0xd3]
          vcvtph2bf8 %ymm3, %xmm2 {%k7}

// CHECK: vcvtph2bf8 %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0xaf,0x74,0xd3]
          vcvtph2bf8 %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8x  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2bf8x  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtph2bf8x  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x7e,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2bf8x  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtph2bf8  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x18,0x74,0x10]
          vcvtph2bf8  (%eax){1to8}, %xmm2

// CHECK: vcvtph2bf8x  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2bf8x  -512(,%ebp,2), %xmm2

// CHECK: vcvtph2bf8x  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0x8f,0x74,0x51,0x7f]
          vcvtph2bf8x  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0x9f,0x74,0x52,0x80]
          vcvtph2bf8  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8  (%eax){1to16}, %xmm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x38,0x74,0x10]
          vcvtph2bf8  (%eax){1to16}, %xmm2

// CHECK: vcvtph2bf8y  -1024(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2bf8y  -1024(,%ebp,2), %xmm2

// CHECK: vcvtph2bf8y  4064(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0xaf,0x74,0x51,0x7f]
          vcvtph2bf8y  4064(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8  -256(%edx){1to16}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0xbf,0x74,0x52,0x80]
          vcvtph2bf8  -256(%edx){1to16}, %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2bf8  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtph2bf8  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf2,0x7e,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2bf8  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtph2bf8  (%eax){1to32}, %ymm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x58,0x74,0x10]
          vcvtph2bf8  (%eax){1to32}, %ymm2

// CHECK: vcvtph2bf8  -2048(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2bf8  -2048(,%ebp,2), %ymm2

// CHECK: vcvtph2bf8  8128(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0xcf,0x74,0x51,0x7f]
          vcvtph2bf8  8128(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtph2bf8  -256(%edx){1to32}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0xdf,0x74,0x52,0x80]
          vcvtph2bf8  -256(%edx){1to32}, %ymm2 {%k7} {z}

// CHECK: vcvtph2bf8s %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x74,0xd3]
          vcvtph2bf8s %xmm3, %xmm2

// CHECK: vcvtph2bf8s %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x74,0xd3]
          vcvtph2bf8s %xmm3, %xmm2 {%k7}

// CHECK: vcvtph2bf8s %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x74,0xd3]
          vcvtph2bf8s %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8s %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x74,0xd3]
          vcvtph2bf8s %zmm3, %ymm2

// CHECK: vcvtph2bf8s %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x74,0xd3]
          vcvtph2bf8s %zmm3, %ymm2 {%k7}

// CHECK: vcvtph2bf8s %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x74,0xd3]
          vcvtph2bf8s %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtph2bf8s %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x74,0xd3]
          vcvtph2bf8s %ymm3, %xmm2

// CHECK: vcvtph2bf8s %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x2f,0x74,0xd3]
          vcvtph2bf8s %ymm3, %xmm2 {%k7}

// CHECK: vcvtph2bf8s %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x74,0xd3]
          vcvtph2bf8s %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8sx  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2bf8sx  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtph2bf8sx  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2bf8sx  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtph2bf8s  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x74,0x10]
          vcvtph2bf8s  (%eax){1to8}, %xmm2

// CHECK: vcvtph2bf8sx  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x74,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2bf8sx  -512(,%ebp,2), %xmm2

// CHECK: vcvtph2bf8sx  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x74,0x51,0x7f]
          vcvtph2bf8sx  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8s  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x9f,0x74,0x52,0x80]
          vcvtph2bf8s  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8s  (%eax){1to16}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x74,0x10]
          vcvtph2bf8s  (%eax){1to16}, %xmm2

// CHECK: vcvtph2bf8sy  -1024(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x74,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2bf8sy  -1024(,%ebp,2), %xmm2

// CHECK: vcvtph2bf8sy  4064(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x74,0x51,0x7f]
          vcvtph2bf8sy  4064(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8s  -256(%edx){1to16}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xbf,0x74,0x52,0x80]
          vcvtph2bf8s  -256(%edx){1to16}, %xmm2 {%k7} {z}

// CHECK: vcvtph2bf8s  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x74,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2bf8s  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtph2bf8s  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x74,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2bf8s  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtph2bf8s  (%eax){1to32}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x74,0x10]
          vcvtph2bf8s  (%eax){1to32}, %ymm2

// CHECK: vcvtph2bf8s  -2048(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x74,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2bf8s  -2048(,%ebp,2), %ymm2

// CHECK: vcvtph2bf8s  8128(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x74,0x51,0x7f]
          vcvtph2bf8s  8128(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtph2bf8s  -256(%edx){1to32}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xdf,0x74,0x52,0x80]
          vcvtph2bf8s  -256(%edx){1to32}, %ymm2 {%k7} {z}

// CHECK: vcvtph2hf8 %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x18,0xd3]
          vcvtph2hf8 %xmm3, %xmm2

// CHECK: vcvtph2hf8 %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x18,0xd3]
          vcvtph2hf8 %xmm3, %xmm2 {%k7}

// CHECK: vcvtph2hf8 %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x18,0xd3]
          vcvtph2hf8 %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8 %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x18,0xd3]
          vcvtph2hf8 %zmm3, %ymm2

// CHECK: vcvtph2hf8 %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x18,0xd3]
          vcvtph2hf8 %zmm3, %ymm2 {%k7}

// CHECK: vcvtph2hf8 %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x18,0xd3]
          vcvtph2hf8 %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtph2hf8 %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x18,0xd3]
          vcvtph2hf8 %ymm3, %xmm2

// CHECK: vcvtph2hf8 %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x2f,0x18,0xd3]
          vcvtph2hf8 %ymm3, %xmm2 {%k7}

// CHECK: vcvtph2hf8 %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x18,0xd3]
          vcvtph2hf8 %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8x  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2hf8x  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtph2hf8x  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2hf8x  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtph2hf8  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x18,0x10]
          vcvtph2hf8  (%eax){1to8}, %xmm2

// CHECK: vcvtph2hf8x  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x18,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2hf8x  -512(,%ebp,2), %xmm2

// CHECK: vcvtph2hf8x  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x18,0x51,0x7f]
          vcvtph2hf8x  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x9f,0x18,0x52,0x80]
          vcvtph2hf8  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8  (%eax){1to16}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x18,0x10]
          vcvtph2hf8  (%eax){1to16}, %xmm2

// CHECK: vcvtph2hf8y  -1024(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x18,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2hf8y  -1024(,%ebp,2), %xmm2

// CHECK: vcvtph2hf8y  4064(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x18,0x51,0x7f]
          vcvtph2hf8y  4064(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8  -256(%edx){1to16}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xbf,0x18,0x52,0x80]
          vcvtph2hf8  -256(%edx){1to16}, %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x18,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2hf8  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtph2hf8  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x18,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2hf8  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtph2hf8  (%eax){1to32}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x18,0x10]
          vcvtph2hf8  (%eax){1to32}, %ymm2

// CHECK: vcvtph2hf8  -2048(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x18,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2hf8  -2048(,%ebp,2), %ymm2

// CHECK: vcvtph2hf8  8128(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x18,0x51,0x7f]
          vcvtph2hf8  8128(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtph2hf8  -256(%edx){1to32}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xdf,0x18,0x52,0x80]
          vcvtph2hf8  -256(%edx){1to32}, %ymm2 {%k7} {z}

// CHECK: vcvtph2hf8s %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x1b,0xd3]
          vcvtph2hf8s %xmm3, %xmm2

// CHECK: vcvtph2hf8s %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x1b,0xd3]
          vcvtph2hf8s %xmm3, %xmm2 {%k7}

// CHECK: vcvtph2hf8s %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x1b,0xd3]
          vcvtph2hf8s %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8s %zmm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x1b,0xd3]
          vcvtph2hf8s %zmm3, %ymm2

// CHECK: vcvtph2hf8s %zmm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x1b,0xd3]
          vcvtph2hf8s %zmm3, %ymm2 {%k7}

// CHECK: vcvtph2hf8s %zmm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x1b,0xd3]
          vcvtph2hf8s %zmm3, %ymm2 {%k7} {z}

// CHECK: vcvtph2hf8s %ymm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x1b,0xd3]
          vcvtph2hf8s %ymm3, %xmm2

// CHECK: vcvtph2hf8s %ymm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x2f,0x1b,0xd3]
          vcvtph2hf8s %ymm3, %xmm2 {%k7}

// CHECK: vcvtph2hf8s %ymm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x1b,0xd3]
          vcvtph2hf8s %ymm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8sx  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2hf8sx  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtph2hf8sx  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x0f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2hf8sx  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtph2hf8s  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x18,0x1b,0x10]
          vcvtph2hf8s  (%eax){1to8}, %xmm2

// CHECK: vcvtph2hf8sx  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x08,0x1b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2hf8sx  -512(,%ebp,2), %xmm2

// CHECK: vcvtph2hf8sx  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x8f,0x1b,0x51,0x7f]
          vcvtph2hf8sx  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8s  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0x9f,0x1b,0x52,0x80]
          vcvtph2hf8s  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8s  (%eax){1to16}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x38,0x1b,0x10]
          vcvtph2hf8s  (%eax){1to16}, %xmm2

// CHECK: vcvtph2hf8sy  -1024(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x28,0x1b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2hf8sy  -1024(,%ebp,2), %xmm2

// CHECK: vcvtph2hf8sy  4064(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xaf,0x1b,0x51,0x7f]
          vcvtph2hf8sy  4064(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8s  -256(%edx){1to16}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xbf,0x1b,0x52,0x80]
          vcvtph2hf8s  -256(%edx){1to16}, %xmm2 {%k7} {z}

// CHECK: vcvtph2hf8s  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x1b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2hf8s  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtph2hf8s  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7e,0x4f,0x1b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2hf8s  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtph2hf8s  (%eax){1to32}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x58,0x1b,0x10]
          vcvtph2hf8s  (%eax){1to32}, %ymm2

// CHECK: vcvtph2hf8s  -2048(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7e,0x48,0x1b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2hf8s  -2048(,%ebp,2), %ymm2

// CHECK: vcvtph2hf8s  8128(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xcf,0x1b,0x51,0x7f]
          vcvtph2hf8s  8128(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtph2hf8s  -256(%edx){1to32}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7e,0xdf,0x1b,0x52,0x80]
          vcvtph2hf8s  -256(%edx){1to32}, %ymm2 {%k7} {z}

