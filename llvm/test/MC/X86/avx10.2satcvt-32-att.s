// RUN: llvm-mc -triple i386 --show-encoding %s | FileCheck %s

// CHECK: vcvtbf162ibs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x69,0xd3]
          vcvtbf162ibs %xmm3, %xmm2

// CHECK: vcvtbf162ibs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x69,0xd3]
          vcvtbf162ibs %xmm3, %xmm2 {%k7}

// CHECK: vcvtbf162ibs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x69,0xd3]
          vcvtbf162ibs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbf162ibs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x69,0xd3]
          vcvtbf162ibs %zmm3, %zmm2

// CHECK: vcvtbf162ibs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x69,0xd3]
          vcvtbf162ibs %zmm3, %zmm2 {%k7}

// CHECK: vcvtbf162ibs %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x69,0xd3]
          vcvtbf162ibs %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvtbf162ibs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x69,0xd3]
          vcvtbf162ibs %ymm3, %ymm2

// CHECK: vcvtbf162ibs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x69,0xd3]
          vcvtbf162ibs %ymm3, %ymm2 {%k7}

// CHECK: vcvtbf162ibs %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x69,0xd3]
          vcvtbf162ibs %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvtbf162ibs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162ibs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtbf162ibs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162ibs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtbf162ibs  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x69,0x10]
          vcvtbf162ibs  (%eax){1to8}, %xmm2

// CHECK: vcvtbf162ibs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x69,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbf162ibs  -512(,%ebp,2), %xmm2

// CHECK: vcvtbf162ibs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x69,0x51,0x7f]
          vcvtbf162ibs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtbf162ibs  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x69,0x52,0x80]
          vcvtbf162ibs  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvtbf162ibs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162ibs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtbf162ibs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162ibs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtbf162ibs  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x69,0x10]
          vcvtbf162ibs  (%eax){1to16}, %ymm2

// CHECK: vcvtbf162ibs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x69,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbf162ibs  -1024(,%ebp,2), %ymm2

// CHECK: vcvtbf162ibs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x69,0x51,0x7f]
          vcvtbf162ibs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtbf162ibs  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x69,0x52,0x80]
          vcvtbf162ibs  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vcvtbf162ibs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162ibs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvtbf162ibs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162ibs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvtbf162ibs  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x69,0x10]
          vcvtbf162ibs  (%eax){1to32}, %zmm2

// CHECK: vcvtbf162ibs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x69,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbf162ibs  -2048(,%ebp,2), %zmm2

// CHECK: vcvtbf162ibs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x69,0x51,0x7f]
          vcvtbf162ibs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvtbf162ibs  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x69,0x52,0x80]
          vcvtbf162ibs  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vcvtbf162iubs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6b,0xd3]
          vcvtbf162iubs %xmm3, %xmm2

// CHECK: vcvtbf162iubs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x6b,0xd3]
          vcvtbf162iubs %xmm3, %xmm2 {%k7}

// CHECK: vcvtbf162iubs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x6b,0xd3]
          vcvtbf162iubs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtbf162iubs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6b,0xd3]
          vcvtbf162iubs %zmm3, %zmm2

// CHECK: vcvtbf162iubs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x6b,0xd3]
          vcvtbf162iubs %zmm3, %zmm2 {%k7}

// CHECK: vcvtbf162iubs %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x6b,0xd3]
          vcvtbf162iubs %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvtbf162iubs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6b,0xd3]
          vcvtbf162iubs %ymm3, %ymm2

// CHECK: vcvtbf162iubs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x6b,0xd3]
          vcvtbf162iubs %ymm3, %ymm2 {%k7}

// CHECK: vcvtbf162iubs %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x6b,0xd3]
          vcvtbf162iubs %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvtbf162iubs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162iubs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtbf162iubs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162iubs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtbf162iubs  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x6b,0x10]
          vcvtbf162iubs  (%eax){1to8}, %xmm2

// CHECK: vcvtbf162iubs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtbf162iubs  -512(,%ebp,2), %xmm2

// CHECK: vcvtbf162iubs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x6b,0x51,0x7f]
          vcvtbf162iubs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtbf162iubs  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x6b,0x52,0x80]
          vcvtbf162iubs  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvtbf162iubs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162iubs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtbf162iubs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162iubs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtbf162iubs  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x6b,0x10]
          vcvtbf162iubs  (%eax){1to16}, %ymm2

// CHECK: vcvtbf162iubs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtbf162iubs  -1024(,%ebp,2), %ymm2

// CHECK: vcvtbf162iubs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x6b,0x51,0x7f]
          vcvtbf162iubs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtbf162iubs  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x6b,0x52,0x80]
          vcvtbf162iubs  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vcvtbf162iubs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtbf162iubs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvtbf162iubs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtbf162iubs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvtbf162iubs  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x6b,0x10]
          vcvtbf162iubs  (%eax){1to32}, %zmm2

// CHECK: vcvtbf162iubs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtbf162iubs  -2048(,%ebp,2), %zmm2

// CHECK: vcvtbf162iubs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x6b,0x51,0x7f]
          vcvtbf162iubs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvtbf162iubs  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x6b,0x52,0x80]
          vcvtbf162iubs  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vcvtph2ibs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x69,0xd3]
          vcvtph2ibs %xmm3, %xmm2

// CHECK: vcvtph2ibs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x69,0xd3]
          vcvtph2ibs %xmm3, %xmm2 {%k7}

// CHECK: vcvtph2ibs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x69,0xd3]
          vcvtph2ibs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2ibs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x69,0xd3]
          vcvtph2ibs %zmm3, %zmm2

// CHECK: vcvtph2ibs {rn-sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x69,0xd3]
          vcvtph2ibs {rn-sae}, %zmm3, %zmm2

// CHECK: vcvtph2ibs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x69,0xd3]
          vcvtph2ibs %zmm3, %zmm2 {%k7}

// CHECK: vcvtph2ibs {rz-sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xff,0x69,0xd3]
          vcvtph2ibs {rz-sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvtph2ibs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x69,0xd3]
          vcvtph2ibs %ymm3, %ymm2

// CHECK: vcvtph2ibs {rn-sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x69,0xd3]
          vcvtph2ibs {rn-sae}, %ymm3, %ymm2

// CHECK: vcvtph2ibs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x69,0xd3]
          vcvtph2ibs %ymm3, %ymm2 {%k7}

// CHECK: vcvtph2ibs {rz-sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x69,0xd3]
          vcvtph2ibs {rz-sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvtph2ibs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2ibs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtph2ibs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2ibs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtph2ibs  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x69,0x10]
          vcvtph2ibs  (%eax){1to8}, %xmm2

// CHECK: vcvtph2ibs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x69,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2ibs  -512(,%ebp,2), %xmm2

// CHECK: vcvtph2ibs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x69,0x51,0x7f]
          vcvtph2ibs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2ibs  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x69,0x52,0x80]
          vcvtph2ibs  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvtph2ibs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2ibs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtph2ibs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2ibs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtph2ibs  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x69,0x10]
          vcvtph2ibs  (%eax){1to16}, %ymm2

// CHECK: vcvtph2ibs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x69,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2ibs  -1024(,%ebp,2), %ymm2

// CHECK: vcvtph2ibs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x69,0x51,0x7f]
          vcvtph2ibs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtph2ibs  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x69,0x52,0x80]
          vcvtph2ibs  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vcvtph2ibs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2ibs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvtph2ibs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2ibs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvtph2ibs  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x69,0x10]
          vcvtph2ibs  (%eax){1to32}, %zmm2

// CHECK: vcvtph2ibs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x69,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2ibs  -2048(,%ebp,2), %zmm2

// CHECK: vcvtph2ibs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x69,0x51,0x7f]
          vcvtph2ibs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvtph2ibs  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x69,0x52,0x80]
          vcvtph2ibs  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vcvtph2iubs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6b,0xd3]
          vcvtph2iubs %xmm3, %xmm2

// CHECK: vcvtph2iubs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6b,0xd3]
          vcvtph2iubs %xmm3, %xmm2 {%k7}

// CHECK: vcvtph2iubs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6b,0xd3]
          vcvtph2iubs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtph2iubs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6b,0xd3]
          vcvtph2iubs %zmm3, %zmm2

// CHECK: vcvtph2iubs {rn-sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6b,0xd3]
          vcvtph2iubs {rn-sae}, %zmm3, %zmm2

// CHECK: vcvtph2iubs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6b,0xd3]
          vcvtph2iubs %zmm3, %zmm2 {%k7}

// CHECK: vcvtph2iubs {rz-sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xff,0x6b,0xd3]
          vcvtph2iubs {rz-sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvtph2iubs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6b,0xd3]
          vcvtph2iubs %ymm3, %ymm2

// CHECK: vcvtph2iubs {rn-sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x6b,0xd3]
          vcvtph2iubs {rn-sae}, %ymm3, %ymm2

// CHECK: vcvtph2iubs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6b,0xd3]
          vcvtph2iubs %ymm3, %ymm2 {%k7}

// CHECK: vcvtph2iubs {rz-sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x78,0xff,0x6b,0xd3]
          vcvtph2iubs {rz-sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvtph2iubs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2iubs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtph2iubs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2iubs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtph2iubs  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6b,0x10]
          vcvtph2iubs  (%eax){1to8}, %xmm2

// CHECK: vcvtph2iubs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtph2iubs  -512(,%ebp,2), %xmm2

// CHECK: vcvtph2iubs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6b,0x51,0x7f]
          vcvtph2iubs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtph2iubs  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6b,0x52,0x80]
          vcvtph2iubs  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvtph2iubs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2iubs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtph2iubs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2iubs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtph2iubs  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x6b,0x10]
          vcvtph2iubs  (%eax){1to16}, %ymm2

// CHECK: vcvtph2iubs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtph2iubs  -1024(,%ebp,2), %ymm2

// CHECK: vcvtph2iubs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x6b,0x51,0x7f]
          vcvtph2iubs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtph2iubs  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x6b,0x52,0x80]
          vcvtph2iubs  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vcvtph2iubs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtph2iubs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvtph2iubs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtph2iubs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvtph2iubs  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x6b,0x10]
          vcvtph2iubs  (%eax){1to32}, %zmm2

// CHECK: vcvtph2iubs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtph2iubs  -2048(,%ebp,2), %zmm2

// CHECK: vcvtph2iubs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x6b,0x51,0x7f]
          vcvtph2iubs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvtph2iubs  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x6b,0x52,0x80]
          vcvtph2iubs  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vcvtps2ibs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x69,0xd3]
          vcvtps2ibs %xmm3, %xmm2

// CHECK: vcvtps2ibs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x69,0xd3]
          vcvtps2ibs %xmm3, %xmm2 {%k7}

// CHECK: vcvtps2ibs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x69,0xd3]
          vcvtps2ibs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtps2ibs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x69,0xd3]
          vcvtps2ibs %zmm3, %zmm2

// CHECK: vcvtps2ibs {rn-sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x69,0xd3]
          vcvtps2ibs {rn-sae}, %zmm3, %zmm2

// CHECK: vcvtps2ibs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x69,0xd3]
          vcvtps2ibs %zmm3, %zmm2 {%k7}

// CHECK: vcvtps2ibs {rz-sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xff,0x69,0xd3]
          vcvtps2ibs {rz-sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvtps2ibs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x69,0xd3]
          vcvtps2ibs %ymm3, %ymm2

// CHECK: vcvtps2ibs {rn-sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x69,0xd3]
          vcvtps2ibs {rn-sae}, %ymm3, %ymm2

// CHECK: vcvtps2ibs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x69,0xd3]
          vcvtps2ibs %ymm3, %ymm2 {%k7}

// CHECK: vcvtps2ibs {rz-sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x69,0xd3]
          vcvtps2ibs {rz-sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvtps2ibs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2ibs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtps2ibs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2ibs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtps2ibs  (%eax){1to4}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x69,0x10]
          vcvtps2ibs  (%eax){1to4}, %xmm2

// CHECK: vcvtps2ibs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x69,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtps2ibs  -512(,%ebp,2), %xmm2

// CHECK: vcvtps2ibs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x69,0x51,0x7f]
          vcvtps2ibs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtps2ibs  -512(%edx){1to4}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x69,0x52,0x80]
          vcvtps2ibs  -512(%edx){1to4}, %xmm2 {%k7} {z}

// CHECK: vcvtps2ibs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2ibs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtps2ibs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2ibs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtps2ibs  (%eax){1to8}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x69,0x10]
          vcvtps2ibs  (%eax){1to8}, %ymm2

// CHECK: vcvtps2ibs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x69,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtps2ibs  -1024(,%ebp,2), %ymm2

// CHECK: vcvtps2ibs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x69,0x51,0x7f]
          vcvtps2ibs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtps2ibs  -512(%edx){1to8}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x69,0x52,0x80]
          vcvtps2ibs  -512(%edx){1to8}, %ymm2 {%k7} {z}

// CHECK: vcvtps2ibs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x69,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2ibs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvtps2ibs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x69,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2ibs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvtps2ibs  (%eax){1to16}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x69,0x10]
          vcvtps2ibs  (%eax){1to16}, %zmm2

// CHECK: vcvtps2ibs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x69,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtps2ibs  -2048(,%ebp,2), %zmm2

// CHECK: vcvtps2ibs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x69,0x51,0x7f]
          vcvtps2ibs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvtps2ibs  -512(%edx){1to16}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x69,0x52,0x80]
          vcvtps2ibs  -512(%edx){1to16}, %zmm2 {%k7} {z}

// CHECK: vcvtps2iubs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6b,0xd3]
          vcvtps2iubs %xmm3, %xmm2

// CHECK: vcvtps2iubs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6b,0xd3]
          vcvtps2iubs %xmm3, %xmm2 {%k7}

// CHECK: vcvtps2iubs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6b,0xd3]
          vcvtps2iubs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvtps2iubs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6b,0xd3]
          vcvtps2iubs %zmm3, %zmm2

// CHECK: vcvtps2iubs {rn-sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6b,0xd3]
          vcvtps2iubs {rn-sae}, %zmm3, %zmm2

// CHECK: vcvtps2iubs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6b,0xd3]
          vcvtps2iubs %zmm3, %zmm2 {%k7}

// CHECK: vcvtps2iubs {rz-sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xff,0x6b,0xd3]
          vcvtps2iubs {rz-sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvtps2iubs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6b,0xd3]
          vcvtps2iubs %ymm3, %ymm2

// CHECK: vcvtps2iubs {rn-sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x6b,0xd3]
          vcvtps2iubs {rn-sae}, %ymm3, %ymm2

// CHECK: vcvtps2iubs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6b,0xd3]
          vcvtps2iubs %ymm3, %ymm2 {%k7}

// CHECK: vcvtps2iubs {rz-sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x79,0xff,0x6b,0xd3]
          vcvtps2iubs {rz-sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvtps2iubs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2iubs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvtps2iubs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2iubs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvtps2iubs  (%eax){1to4}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6b,0x10]
          vcvtps2iubs  (%eax){1to4}, %xmm2

// CHECK: vcvtps2iubs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6b,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvtps2iubs  -512(,%ebp,2), %xmm2

// CHECK: vcvtps2iubs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6b,0x51,0x7f]
          vcvtps2iubs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvtps2iubs  -512(%edx){1to4}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6b,0x52,0x80]
          vcvtps2iubs  -512(%edx){1to4}, %xmm2 {%k7} {z}

// CHECK: vcvtps2iubs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2iubs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvtps2iubs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2iubs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvtps2iubs  (%eax){1to8}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x6b,0x10]
          vcvtps2iubs  (%eax){1to8}, %ymm2

// CHECK: vcvtps2iubs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6b,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvtps2iubs  -1024(,%ebp,2), %ymm2

// CHECK: vcvtps2iubs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x6b,0x51,0x7f]
          vcvtps2iubs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvtps2iubs  -512(%edx){1to8}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x6b,0x52,0x80]
          vcvtps2iubs  -512(%edx){1to8}, %ymm2 {%k7} {z}

// CHECK: vcvtps2iubs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6b,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvtps2iubs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvtps2iubs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6b,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvtps2iubs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvtps2iubs  (%eax){1to16}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x6b,0x10]
          vcvtps2iubs  (%eax){1to16}, %zmm2

// CHECK: vcvtps2iubs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6b,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvtps2iubs  -2048(,%ebp,2), %zmm2

// CHECK: vcvtps2iubs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x6b,0x51,0x7f]
          vcvtps2iubs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvtps2iubs  -512(%edx){1to16}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x6b,0x52,0x80]
          vcvtps2iubs  -512(%edx){1to16}, %zmm2 {%k7} {z}

// CHECK: vcvttbf162ibs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x68,0xd3]
          vcvttbf162ibs %xmm3, %xmm2

// CHECK: vcvttbf162ibs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x68,0xd3]
          vcvttbf162ibs %xmm3, %xmm2 {%k7}

// CHECK: vcvttbf162ibs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x68,0xd3]
          vcvttbf162ibs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttbf162ibs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x68,0xd3]
          vcvttbf162ibs %zmm3, %zmm2

// CHECK: vcvttbf162ibs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x68,0xd3]
          vcvttbf162ibs %zmm3, %zmm2 {%k7}

// CHECK: vcvttbf162ibs %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x68,0xd3]
          vcvttbf162ibs %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttbf162ibs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x68,0xd3]
          vcvttbf162ibs %ymm3, %ymm2

// CHECK: vcvttbf162ibs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x68,0xd3]
          vcvttbf162ibs %ymm3, %ymm2 {%k7}

// CHECK: vcvttbf162ibs %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x68,0xd3]
          vcvttbf162ibs %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttbf162ibs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162ibs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttbf162ibs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162ibs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttbf162ibs  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x68,0x10]
          vcvttbf162ibs  (%eax){1to8}, %xmm2

// CHECK: vcvttbf162ibs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x68,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttbf162ibs  -512(,%ebp,2), %xmm2

// CHECK: vcvttbf162ibs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x68,0x51,0x7f]
          vcvttbf162ibs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttbf162ibs  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x68,0x52,0x80]
          vcvttbf162ibs  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvttbf162ibs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162ibs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttbf162ibs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162ibs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttbf162ibs  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x68,0x10]
          vcvttbf162ibs  (%eax){1to16}, %ymm2

// CHECK: vcvttbf162ibs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x68,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttbf162ibs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttbf162ibs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x68,0x51,0x7f]
          vcvttbf162ibs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttbf162ibs  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x68,0x52,0x80]
          vcvttbf162ibs  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vcvttbf162ibs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162ibs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttbf162ibs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162ibs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttbf162ibs  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x68,0x10]
          vcvttbf162ibs  (%eax){1to32}, %zmm2

// CHECK: vcvttbf162ibs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x68,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttbf162ibs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttbf162ibs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x68,0x51,0x7f]
          vcvttbf162ibs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttbf162ibs  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x68,0x52,0x80]
          vcvttbf162ibs  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vcvttbf162iubs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6a,0xd3]
          vcvttbf162iubs %xmm3, %xmm2

// CHECK: vcvttbf162iubs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x6a,0xd3]
          vcvttbf162iubs %xmm3, %xmm2 {%k7}

// CHECK: vcvttbf162iubs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x6a,0xd3]
          vcvttbf162iubs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttbf162iubs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6a,0xd3]
          vcvttbf162iubs %zmm3, %zmm2

// CHECK: vcvttbf162iubs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x6a,0xd3]
          vcvttbf162iubs %zmm3, %zmm2 {%k7}

// CHECK: vcvttbf162iubs %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x6a,0xd3]
          vcvttbf162iubs %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttbf162iubs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6a,0xd3]
          vcvttbf162iubs %ymm3, %ymm2

// CHECK: vcvttbf162iubs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x6a,0xd3]
          vcvttbf162iubs %ymm3, %ymm2 {%k7}

// CHECK: vcvttbf162iubs %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x6a,0xd3]
          vcvttbf162iubs %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttbf162iubs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162iubs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttbf162iubs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x0f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162iubs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttbf162iubs  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x18,0x6a,0x10]
          vcvttbf162iubs  (%eax){1to8}, %xmm2

// CHECK: vcvttbf162iubs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x08,0x6a,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttbf162iubs  -512(,%ebp,2), %xmm2

// CHECK: vcvttbf162iubs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x8f,0x6a,0x51,0x7f]
          vcvttbf162iubs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttbf162iubs  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0x9f,0x6a,0x52,0x80]
          vcvttbf162iubs  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvttbf162iubs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162iubs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttbf162iubs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x2f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162iubs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttbf162iubs  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x38,0x6a,0x10]
          vcvttbf162iubs  (%eax){1to16}, %ymm2

// CHECK: vcvttbf162iubs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x28,0x6a,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttbf162iubs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttbf162iubs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xaf,0x6a,0x51,0x7f]
          vcvttbf162iubs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttbf162iubs  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xbf,0x6a,0x52,0x80]
          vcvttbf162iubs  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vcvttbf162iubs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttbf162iubs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttbf162iubs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7f,0x4f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttbf162iubs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttbf162iubs  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x58,0x6a,0x10]
          vcvttbf162iubs  (%eax){1to32}, %zmm2

// CHECK: vcvttbf162iubs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7f,0x48,0x6a,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttbf162iubs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttbf162iubs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xcf,0x6a,0x51,0x7f]
          vcvttbf162iubs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttbf162iubs  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7f,0xdf,0x6a,0x52,0x80]
          vcvttbf162iubs  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vcvttph2ibs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x68,0xd3]
          vcvttph2ibs %xmm3, %xmm2

// CHECK: vcvttph2ibs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x68,0xd3]
          vcvttph2ibs %xmm3, %xmm2 {%k7}

// CHECK: vcvttph2ibs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x68,0xd3]
          vcvttph2ibs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttph2ibs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x68,0xd3]
          vcvttph2ibs %zmm3, %zmm2

// CHECK: vcvttph2ibs {sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x68,0xd3]
          vcvttph2ibs {sae}, %zmm3, %zmm2

// CHECK: vcvttph2ibs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x68,0xd3]
          vcvttph2ibs %zmm3, %zmm2 {%k7}

// CHECK: vcvttph2ibs {sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x68,0xd3]
          vcvttph2ibs {sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttph2ibs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x68,0xd3]
          vcvttph2ibs %ymm3, %ymm2

// CHECK: vcvttph2ibs {sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x68,0xd3]
          vcvttph2ibs {sae}, %ymm3, %ymm2

// CHECK: vcvttph2ibs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x68,0xd3]
          vcvttph2ibs %ymm3, %ymm2 {%k7}

// CHECK: vcvttph2ibs {sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x68,0xd3]
          vcvttph2ibs {sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttph2ibs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2ibs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttph2ibs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2ibs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttph2ibs  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x68,0x10]
          vcvttph2ibs  (%eax){1to8}, %xmm2

// CHECK: vcvttph2ibs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x68,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttph2ibs  -512(,%ebp,2), %xmm2

// CHECK: vcvttph2ibs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x68,0x51,0x7f]
          vcvttph2ibs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttph2ibs  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x68,0x52,0x80]
          vcvttph2ibs  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvttph2ibs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2ibs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttph2ibs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2ibs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttph2ibs  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x68,0x10]
          vcvttph2ibs  (%eax){1to16}, %ymm2

// CHECK: vcvttph2ibs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x68,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttph2ibs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttph2ibs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x68,0x51,0x7f]
          vcvttph2ibs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttph2ibs  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x68,0x52,0x80]
          vcvttph2ibs  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vcvttph2ibs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2ibs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttph2ibs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2ibs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttph2ibs  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x68,0x10]
          vcvttph2ibs  (%eax){1to32}, %zmm2

// CHECK: vcvttph2ibs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x68,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttph2ibs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttph2ibs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x68,0x51,0x7f]
          vcvttph2ibs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttph2ibs  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x68,0x52,0x80]
          vcvttph2ibs  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vcvttph2iubs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6a,0xd3]
          vcvttph2iubs %xmm3, %xmm2

// CHECK: vcvttph2iubs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6a,0xd3]
          vcvttph2iubs %xmm3, %xmm2 {%k7}

// CHECK: vcvttph2iubs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6a,0xd3]
          vcvttph2iubs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttph2iubs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6a,0xd3]
          vcvttph2iubs %zmm3, %zmm2

// CHECK: vcvttph2iubs {sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6a,0xd3]
          vcvttph2iubs {sae}, %zmm3, %zmm2

// CHECK: vcvttph2iubs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6a,0xd3]
          vcvttph2iubs %zmm3, %zmm2 {%k7}

// CHECK: vcvttph2iubs {sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6a,0xd3]
          vcvttph2iubs {sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttph2iubs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6a,0xd3]
          vcvttph2iubs %ymm3, %ymm2

// CHECK: vcvttph2iubs {sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x78,0x18,0x6a,0xd3]
          vcvttph2iubs {sae}, %ymm3, %ymm2

// CHECK: vcvttph2iubs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6a,0xd3]
          vcvttph2iubs %ymm3, %ymm2 {%k7}

// CHECK: vcvttph2iubs {sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x78,0x9f,0x6a,0xd3]
          vcvttph2iubs {sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttph2iubs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2iubs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttph2iubs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x0f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2iubs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttph2iubs  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x18,0x6a,0x10]
          vcvttph2iubs  (%eax){1to8}, %xmm2

// CHECK: vcvttph2iubs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x08,0x6a,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttph2iubs  -512(,%ebp,2), %xmm2

// CHECK: vcvttph2iubs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x8f,0x6a,0x51,0x7f]
          vcvttph2iubs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttph2iubs  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0x9f,0x6a,0x52,0x80]
          vcvttph2iubs  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vcvttph2iubs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2iubs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttph2iubs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x2f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2iubs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttph2iubs  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x38,0x6a,0x10]
          vcvttph2iubs  (%eax){1to16}, %ymm2

// CHECK: vcvttph2iubs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x28,0x6a,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttph2iubs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttph2iubs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xaf,0x6a,0x51,0x7f]
          vcvttph2iubs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttph2iubs  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xbf,0x6a,0x52,0x80]
          vcvttph2iubs  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vcvttph2iubs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttph2iubs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttph2iubs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7c,0x4f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttph2iubs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttph2iubs  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x58,0x6a,0x10]
          vcvttph2iubs  (%eax){1to32}, %zmm2

// CHECK: vcvttph2iubs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7c,0x48,0x6a,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttph2iubs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttph2iubs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xcf,0x6a,0x51,0x7f]
          vcvttph2iubs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttph2iubs  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7c,0xdf,0x6a,0x52,0x80]
          vcvttph2iubs  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vcvttps2ibs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x68,0xd3]
          vcvttps2ibs %xmm3, %xmm2

// CHECK: vcvttps2ibs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x68,0xd3]
          vcvttps2ibs %xmm3, %xmm2 {%k7}

// CHECK: vcvttps2ibs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x68,0xd3]
          vcvttps2ibs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttps2ibs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x68,0xd3]
          vcvttps2ibs %zmm3, %zmm2

// CHECK: vcvttps2ibs {sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x68,0xd3]
          vcvttps2ibs {sae}, %zmm3, %zmm2

// CHECK: vcvttps2ibs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x68,0xd3]
          vcvttps2ibs %zmm3, %zmm2 {%k7}

// CHECK: vcvttps2ibs {sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x68,0xd3]
          vcvttps2ibs {sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttps2ibs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x68,0xd3]
          vcvttps2ibs %ymm3, %ymm2

// CHECK: vcvttps2ibs {sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x68,0xd3]
          vcvttps2ibs {sae}, %ymm3, %ymm2

// CHECK: vcvttps2ibs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x68,0xd3]
          vcvttps2ibs %ymm3, %ymm2 {%k7}

// CHECK: vcvttps2ibs {sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x68,0xd3]
          vcvttps2ibs {sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttps2ibs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2ibs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttps2ibs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2ibs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttps2ibs  (%eax){1to4}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x68,0x10]
          vcvttps2ibs  (%eax){1to4}, %xmm2

// CHECK: vcvttps2ibs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x68,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2ibs  -512(,%ebp,2), %xmm2

// CHECK: vcvttps2ibs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x68,0x51,0x7f]
          vcvttps2ibs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttps2ibs  -512(%edx){1to4}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x68,0x52,0x80]
          vcvttps2ibs  -512(%edx){1to4}, %xmm2 {%k7} {z}

// CHECK: vcvttps2ibs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2ibs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttps2ibs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2ibs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttps2ibs  (%eax){1to8}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x68,0x10]
          vcvttps2ibs  (%eax){1to8}, %ymm2

// CHECK: vcvttps2ibs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x68,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2ibs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttps2ibs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x68,0x51,0x7f]
          vcvttps2ibs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttps2ibs  -512(%edx){1to8}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x68,0x52,0x80]
          vcvttps2ibs  -512(%edx){1to8}, %ymm2 {%k7} {z}

// CHECK: vcvttps2ibs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x68,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2ibs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttps2ibs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x68,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2ibs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttps2ibs  (%eax){1to16}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x68,0x10]
          vcvttps2ibs  (%eax){1to16}, %zmm2

// CHECK: vcvttps2ibs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x68,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2ibs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttps2ibs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x68,0x51,0x7f]
          vcvttps2ibs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttps2ibs  -512(%edx){1to16}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x68,0x52,0x80]
          vcvttps2ibs  -512(%edx){1to16}, %zmm2 {%k7} {z}

// CHECK: vcvttps2iubs %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6a,0xd3]
          vcvttps2iubs %xmm3, %xmm2

// CHECK: vcvttps2iubs %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6a,0xd3]
          vcvttps2iubs %xmm3, %xmm2 {%k7}

// CHECK: vcvttps2iubs %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6a,0xd3]
          vcvttps2iubs %xmm3, %xmm2 {%k7} {z}

// CHECK: vcvttps2iubs %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6a,0xd3]
          vcvttps2iubs %zmm3, %zmm2

// CHECK: vcvttps2iubs {sae}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6a,0xd3]
          vcvttps2iubs {sae}, %zmm3, %zmm2

// CHECK: vcvttps2iubs %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6a,0xd3]
          vcvttps2iubs %zmm3, %zmm2 {%k7}

// CHECK: vcvttps2iubs {sae}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6a,0xd3]
          vcvttps2iubs {sae}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vcvttps2iubs %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6a,0xd3]
          vcvttps2iubs %ymm3, %ymm2

// CHECK: vcvttps2iubs {sae}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x79,0x18,0x6a,0xd3]
          vcvttps2iubs {sae}, %ymm3, %ymm2

// CHECK: vcvttps2iubs %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6a,0xd3]
          vcvttps2iubs %ymm3, %ymm2 {%k7}

// CHECK: vcvttps2iubs {sae}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x79,0x9f,0x6a,0xd3]
          vcvttps2iubs {sae}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vcvttps2iubs  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2iubs  268435456(%esp,%esi,8), %xmm2

// CHECK: vcvttps2iubs  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2iubs  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vcvttps2iubs  (%eax){1to4}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x6a,0x10]
          vcvttps2iubs  (%eax){1to4}, %xmm2

// CHECK: vcvttps2iubs  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x6a,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vcvttps2iubs  -512(,%ebp,2), %xmm2

// CHECK: vcvttps2iubs  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x6a,0x51,0x7f]
          vcvttps2iubs  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vcvttps2iubs  -512(%edx){1to4}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x6a,0x52,0x80]
          vcvttps2iubs  -512(%edx){1to4}, %xmm2 {%k7} {z}

// CHECK: vcvttps2iubs  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2iubs  268435456(%esp,%esi,8), %ymm2

// CHECK: vcvttps2iubs  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2iubs  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vcvttps2iubs  (%eax){1to8}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x6a,0x10]
          vcvttps2iubs  (%eax){1to8}, %ymm2

// CHECK: vcvttps2iubs  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x6a,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vcvttps2iubs  -1024(,%ebp,2), %ymm2

// CHECK: vcvttps2iubs  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x6a,0x51,0x7f]
          vcvttps2iubs  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vcvttps2iubs  -512(%edx){1to8}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x6a,0x52,0x80]
          vcvttps2iubs  -512(%edx){1to8}, %ymm2 {%k7} {z}

// CHECK: vcvttps2iubs  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcvttps2iubs  268435456(%esp,%esi,8), %zmm2

// CHECK: vcvttps2iubs  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x6a,0x94,0x87,0x23,0x01,0x00,0x00]
          vcvttps2iubs  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vcvttps2iubs  (%eax){1to16}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x6a,0x10]
          vcvttps2iubs  (%eax){1to16}, %zmm2

// CHECK: vcvttps2iubs  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x6a,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vcvttps2iubs  -2048(,%ebp,2), %zmm2

// CHECK: vcvttps2iubs  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x6a,0x51,0x7f]
          vcvttps2iubs  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vcvttps2iubs  -512(%edx){1to16}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x6a,0x52,0x80]
          vcvttps2iubs  -512(%edx){1to16}, %zmm2 {%k7} {z}

