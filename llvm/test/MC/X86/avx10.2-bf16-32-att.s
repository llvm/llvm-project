// RUN: llvm-mc -triple i386 --show-encoding %s | FileCheck %s

// CHECK: vaddbf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x58,0xd4]
          vaddbf16 %ymm4, %ymm3, %ymm2

// CHECK: vaddbf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x58,0xd4]
          vaddbf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vaddbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x58,0xd4]
          vaddbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vaddbf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x58,0xd4]
          vaddbf16 %zmm4, %zmm3, %zmm2

// CHECK: vaddbf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x58,0xd4]
          vaddbf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vaddbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x58,0xd4]
          vaddbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vaddbf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x58,0xd4]
          vaddbf16 %xmm4, %xmm3, %xmm2

// CHECK: vaddbf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x58,0xd4]
          vaddbf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vaddbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x58,0xd4]
          vaddbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vaddbf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x58,0x94,0xf4,0x00,0x00,0x00,0x10]
          vaddbf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vaddbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x58,0x94,0x87,0x23,0x01,0x00,0x00]
          vaddbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vaddbf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x58,0x10]
          vaddbf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vaddbf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x58,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vaddbf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vaddbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x58,0x51,0x7f]
          vaddbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vaddbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x58,0x52,0x80]
          vaddbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vaddbf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x58,0x94,0xf4,0x00,0x00,0x00,0x10]
          vaddbf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vaddbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x58,0x94,0x87,0x23,0x01,0x00,0x00]
          vaddbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vaddbf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x58,0x10]
          vaddbf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vaddbf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x58,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vaddbf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vaddbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x58,0x51,0x7f]
          vaddbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vaddbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x58,0x52,0x80]
          vaddbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vaddbf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x58,0x94,0xf4,0x00,0x00,0x00,0x10]
          vaddbf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vaddbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x58,0x94,0x87,0x23,0x01,0x00,0x00]
          vaddbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vaddbf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x58,0x10]
          vaddbf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vaddbf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x58,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vaddbf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vaddbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x58,0x51,0x7f]
          vaddbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vaddbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x58,0x52,0x80]
          vaddbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vcmpbf16 $123, %ymm4, %ymm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0xc2,0xec,0x7b]
          vcmpbf16 $123, %ymm4, %ymm3, %k5

// CHECK: vcmpbf16 $123, %ymm4, %ymm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0xc2,0xec,0x7b]
          vcmpbf16 $123, %ymm4, %ymm3, %k5 {%k7}

// CHECK: vcmpbf16 $123, %xmm4, %xmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0xc2,0xec,0x7b]
          vcmpbf16 $123, %xmm4, %xmm3, %k5

// CHECK: vcmpbf16 $123, %xmm4, %xmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0xc2,0xec,0x7b]
          vcmpbf16 $123, %xmm4, %xmm3, %k5 {%k7}

// CHECK: vcmpbf16 $123, %zmm4, %zmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0xc2,0xec,0x7b]
          vcmpbf16 $123, %zmm4, %zmm3, %k5

// CHECK: vcmpbf16 $123, %zmm4, %zmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0xc2,0xec,0x7b]
          vcmpbf16 $123, %zmm4, %zmm3, %k5 {%k7}

// CHECK: vcmpbf16  $123, 268435456(%esp,%esi,8), %zmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16  $123, 268435456(%esp,%esi,8), %zmm3, %k5

// CHECK: vcmpbf16  $123, 291(%edi,%eax,4), %zmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0xc2,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16  $123, 291(%edi,%eax,4), %zmm3, %k5 {%k7}

// CHECK: vcmpbf16  $123, (%eax){1to32}, %zmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x58,0xc2,0x28,0x7b]
          vcmpbf16  $123, (%eax){1to32}, %zmm3, %k5

// CHECK: vcmpbf16  $123, -2048(,%ebp,2), %zmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0xc2,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vcmpbf16  $123, -2048(,%ebp,2), %zmm3, %k5

// CHECK: vcmpbf16  $123, 8128(%ecx), %zmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0xc2,0x69,0x7f,0x7b]
          vcmpbf16  $123, 8128(%ecx), %zmm3, %k5 {%k7}

// CHECK: vcmpbf16  $123, -256(%edx){1to32}, %zmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x5f,0xc2,0x6a,0x80,0x7b]
          vcmpbf16  $123, -256(%edx){1to32}, %zmm3, %k5 {%k7}

// CHECK: vcmpbf16  $123, 268435456(%esp,%esi,8), %xmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16  $123, 268435456(%esp,%esi,8), %xmm3, %k5

// CHECK: vcmpbf16  $123, 291(%edi,%eax,4), %xmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0xc2,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16  $123, 291(%edi,%eax,4), %xmm3, %k5 {%k7}

// CHECK: vcmpbf16  $123, (%eax){1to8}, %xmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x18,0xc2,0x28,0x7b]
          vcmpbf16  $123, (%eax){1to8}, %xmm3, %k5

// CHECK: vcmpbf16  $123, -512(,%ebp,2), %xmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0xc2,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vcmpbf16  $123, -512(,%ebp,2), %xmm3, %k5

// CHECK: vcmpbf16  $123, 2032(%ecx), %xmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0xc2,0x69,0x7f,0x7b]
          vcmpbf16  $123, 2032(%ecx), %xmm3, %k5 {%k7}

// CHECK: vcmpbf16  $123, -256(%edx){1to8}, %xmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x1f,0xc2,0x6a,0x80,0x7b]
          vcmpbf16  $123, -256(%edx){1to8}, %xmm3, %k5 {%k7}

// CHECK: vcmpbf16  $123, 268435456(%esp,%esi,8), %ymm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0xc2,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vcmpbf16  $123, 268435456(%esp,%esi,8), %ymm3, %k5

// CHECK: vcmpbf16  $123, 291(%edi,%eax,4), %ymm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0xc2,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vcmpbf16  $123, 291(%edi,%eax,4), %ymm3, %k5 {%k7}

// CHECK: vcmpbf16  $123, (%eax){1to16}, %ymm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x38,0xc2,0x28,0x7b]
          vcmpbf16  $123, (%eax){1to16}, %ymm3, %k5

// CHECK: vcmpbf16  $123, -1024(,%ebp,2), %ymm3, %k5
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0xc2,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vcmpbf16  $123, -1024(,%ebp,2), %ymm3, %k5

// CHECK: vcmpbf16  $123, 4064(%ecx), %ymm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0xc2,0x69,0x7f,0x7b]
          vcmpbf16  $123, 4064(%ecx), %ymm3, %k5 {%k7}

// CHECK: vcmpbf16  $123, -256(%edx){1to16}, %ymm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x3f,0xc2,0x6a,0x80,0x7b]
          vcmpbf16  $123, -256(%edx){1to16}, %ymm3, %k5 {%k7}

// CHECK: vcomisbf16 %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0xd3]
          vcomisbf16 %xmm3, %xmm2

// CHECK: vcomisbf16  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vcomisbf16  268435456(%esp,%esi,8), %xmm2

// CHECK: vcomisbf16  291(%edi,%eax,4), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x94,0x87,0x23,0x01,0x00,0x00]
          vcomisbf16  291(%edi,%eax,4), %xmm2

// CHECK: vcomisbf16  (%eax), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x10]
          vcomisbf16  (%eax), %xmm2

// CHECK: vcomisbf16  -64(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x14,0x6d,0xc0,0xff,0xff,0xff]
          vcomisbf16  -64(,%ebp,2), %xmm2

// CHECK: vcomisbf16  254(%ecx), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x51,0x7f]
          vcomisbf16  254(%ecx), %xmm2

// CHECK: vcomisbf16  -256(%edx), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x2f,0x52,0x80]
          vcomisbf16  -256(%edx), %xmm2

// CHECK: vdivbf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5e,0xd4]
          vdivbf16 %ymm4, %ymm3, %ymm2

// CHECK: vdivbf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5e,0xd4]
          vdivbf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vdivbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5e,0xd4]
          vdivbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vdivbf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5e,0xd4]
          vdivbf16 %zmm4, %zmm3, %zmm2

// CHECK: vdivbf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5e,0xd4]
          vdivbf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vdivbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5e,0xd4]
          vdivbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vdivbf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5e,0xd4]
          vdivbf16 %xmm4, %xmm3, %xmm2

// CHECK: vdivbf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5e,0xd4]
          vdivbf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vdivbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5e,0xd4]
          vdivbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vdivbf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdivbf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vdivbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5e,0x94,0x87,0x23,0x01,0x00,0x00]
          vdivbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vdivbf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5e,0x10]
          vdivbf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vdivbf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5e,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vdivbf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vdivbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5e,0x51,0x7f]
          vdivbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vdivbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5e,0x52,0x80]
          vdivbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vdivbf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdivbf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vdivbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5e,0x94,0x87,0x23,0x01,0x00,0x00]
          vdivbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vdivbf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5e,0x10]
          vdivbf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vdivbf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vdivbf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vdivbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5e,0x51,0x7f]
          vdivbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vdivbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5e,0x52,0x80]
          vdivbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vdivbf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vdivbf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vdivbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5e,0x94,0x87,0x23,0x01,0x00,0x00]
          vdivbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vdivbf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5e,0x10]
          vdivbf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vdivbf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vdivbf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vdivbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5e,0x51,0x7f]
          vdivbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vdivbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5e,0x52,0x80]
          vdivbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmadd132bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x98,0xd4]
          vfmadd132bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfmadd132bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x98,0xd4]
          vfmadd132bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfmadd132bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x98,0xd4]
          vfmadd132bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmadd132bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x98,0xd4]
          vfmadd132bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfmadd132bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x98,0xd4]
          vfmadd132bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfmadd132bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x98,0xd4]
          vfmadd132bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmadd132bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x98,0xd4]
          vfmadd132bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfmadd132bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x98,0xd4]
          vfmadd132bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfmadd132bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x98,0xd4]
          vfmadd132bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmadd132bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x98,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfmadd132bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x98,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd132bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfmadd132bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x98,0x10]
          vfmadd132bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfmadd132bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x98,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd132bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfmadd132bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x98,0x51,0x7f]
          vfmadd132bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmadd132bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x98,0x52,0x80]
          vfmadd132bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmadd132bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x98,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfmadd132bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x98,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd132bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfmadd132bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x98,0x10]
          vfmadd132bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfmadd132bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x98,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd132bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfmadd132bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x98,0x51,0x7f]
          vfmadd132bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmadd132bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x98,0x52,0x80]
          vfmadd132bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmadd132bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x98,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd132bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfmadd132bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x98,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd132bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfmadd132bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x98,0x10]
          vfmadd132bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfmadd132bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x98,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd132bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfmadd132bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x98,0x51,0x7f]
          vfmadd132bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmadd132bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x98,0x52,0x80]
          vfmadd132bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmadd213bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xa8,0xd4]
          vfmadd213bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfmadd213bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xa8,0xd4]
          vfmadd213bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfmadd213bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xa8,0xd4]
          vfmadd213bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmadd213bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xa8,0xd4]
          vfmadd213bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfmadd213bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xa8,0xd4]
          vfmadd213bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfmadd213bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xa8,0xd4]
          vfmadd213bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmadd213bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xa8,0xd4]
          vfmadd213bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfmadd213bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xa8,0xd4]
          vfmadd213bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfmadd213bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xa8,0xd4]
          vfmadd213bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmadd213bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xa8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfmadd213bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xa8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd213bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfmadd213bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xa8,0x10]
          vfmadd213bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfmadd213bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xa8,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd213bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfmadd213bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xa8,0x51,0x7f]
          vfmadd213bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmadd213bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xa8,0x52,0x80]
          vfmadd213bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmadd213bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xa8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfmadd213bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xa8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd213bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfmadd213bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xa8,0x10]
          vfmadd213bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfmadd213bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xa8,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd213bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfmadd213bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xa8,0x51,0x7f]
          vfmadd213bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmadd213bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xa8,0x52,0x80]
          vfmadd213bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmadd213bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xa8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd213bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfmadd213bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xa8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd213bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfmadd213bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xa8,0x10]
          vfmadd213bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfmadd213bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xa8,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd213bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfmadd213bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xa8,0x51,0x7f]
          vfmadd213bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmadd213bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xa8,0x52,0x80]
          vfmadd213bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmadd231bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xb8,0xd4]
          vfmadd231bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfmadd231bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xb8,0xd4]
          vfmadd231bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfmadd231bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xb8,0xd4]
          vfmadd231bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmadd231bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xb8,0xd4]
          vfmadd231bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfmadd231bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xb8,0xd4]
          vfmadd231bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfmadd231bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xb8,0xd4]
          vfmadd231bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmadd231bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xb8,0xd4]
          vfmadd231bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfmadd231bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xb8,0xd4]
          vfmadd231bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfmadd231bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xb8,0xd4]
          vfmadd231bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmadd231bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xb8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfmadd231bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xb8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd231bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfmadd231bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xb8,0x10]
          vfmadd231bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfmadd231bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xb8,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmadd231bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfmadd231bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xb8,0x51,0x7f]
          vfmadd231bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmadd231bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xb8,0x52,0x80]
          vfmadd231bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmadd231bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xb8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfmadd231bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xb8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd231bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfmadd231bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xb8,0x10]
          vfmadd231bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfmadd231bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xb8,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmadd231bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfmadd231bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xb8,0x51,0x7f]
          vfmadd231bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmadd231bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xb8,0x52,0x80]
          vfmadd231bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmadd231bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xb8,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmadd231bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfmadd231bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xb8,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmadd231bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfmadd231bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xb8,0x10]
          vfmadd231bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfmadd231bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xb8,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmadd231bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfmadd231bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xb8,0x51,0x7f]
          vfmadd231bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmadd231bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xb8,0x52,0x80]
          vfmadd231bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmsub132bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9a,0xd4]
          vfmsub132bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfmsub132bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9a,0xd4]
          vfmsub132bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfmsub132bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9a,0xd4]
          vfmsub132bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmsub132bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9a,0xd4]
          vfmsub132bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfmsub132bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9a,0xd4]
          vfmsub132bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfmsub132bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9a,0xd4]
          vfmsub132bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmsub132bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9a,0xd4]
          vfmsub132bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfmsub132bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9a,0xd4]
          vfmsub132bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfmsub132bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9a,0xd4]
          vfmsub132bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmsub132bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfmsub132bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9a,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub132bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfmsub132bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x9a,0x10]
          vfmsub132bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfmsub132bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9a,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub132bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfmsub132bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9a,0x51,0x7f]
          vfmsub132bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmsub132bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x9a,0x52,0x80]
          vfmsub132bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmsub132bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfmsub132bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9a,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub132bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfmsub132bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x9a,0x10]
          vfmsub132bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfmsub132bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9a,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub132bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfmsub132bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9a,0x51,0x7f]
          vfmsub132bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmsub132bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x9a,0x52,0x80]
          vfmsub132bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmsub132bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9a,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub132bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfmsub132bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9a,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub132bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfmsub132bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x9a,0x10]
          vfmsub132bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfmsub132bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9a,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub132bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfmsub132bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9a,0x51,0x7f]
          vfmsub132bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmsub132bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x9a,0x52,0x80]
          vfmsub132bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmsub213bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xaa,0xd4]
          vfmsub213bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfmsub213bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xaa,0xd4]
          vfmsub213bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfmsub213bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xaa,0xd4]
          vfmsub213bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmsub213bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xaa,0xd4]
          vfmsub213bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfmsub213bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xaa,0xd4]
          vfmsub213bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfmsub213bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xaa,0xd4]
          vfmsub213bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmsub213bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xaa,0xd4]
          vfmsub213bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfmsub213bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xaa,0xd4]
          vfmsub213bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfmsub213bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xaa,0xd4]
          vfmsub213bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmsub213bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xaa,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfmsub213bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xaa,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub213bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfmsub213bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xaa,0x10]
          vfmsub213bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfmsub213bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xaa,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub213bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfmsub213bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xaa,0x51,0x7f]
          vfmsub213bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmsub213bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xaa,0x52,0x80]
          vfmsub213bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmsub213bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xaa,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfmsub213bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xaa,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub213bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfmsub213bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xaa,0x10]
          vfmsub213bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfmsub213bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xaa,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub213bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfmsub213bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xaa,0x51,0x7f]
          vfmsub213bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmsub213bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xaa,0x52,0x80]
          vfmsub213bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmsub213bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xaa,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub213bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfmsub213bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xaa,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub213bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfmsub213bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xaa,0x10]
          vfmsub213bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfmsub213bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xaa,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub213bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfmsub213bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xaa,0x51,0x7f]
          vfmsub213bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmsub213bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xaa,0x52,0x80]
          vfmsub213bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmsub231bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xba,0xd4]
          vfmsub231bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfmsub231bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xba,0xd4]
          vfmsub231bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfmsub231bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xba,0xd4]
          vfmsub231bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmsub231bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xba,0xd4]
          vfmsub231bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfmsub231bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xba,0xd4]
          vfmsub231bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfmsub231bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xba,0xd4]
          vfmsub231bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmsub231bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xba,0xd4]
          vfmsub231bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfmsub231bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xba,0xd4]
          vfmsub231bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfmsub231bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xba,0xd4]
          vfmsub231bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmsub231bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xba,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfmsub231bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xba,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub231bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfmsub231bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xba,0x10]
          vfmsub231bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfmsub231bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xba,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfmsub231bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfmsub231bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xba,0x51,0x7f]
          vfmsub231bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmsub231bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xba,0x52,0x80]
          vfmsub231bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfmsub231bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xba,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfmsub231bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xba,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub231bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfmsub231bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xba,0x10]
          vfmsub231bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfmsub231bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xba,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfmsub231bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfmsub231bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xba,0x51,0x7f]
          vfmsub231bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmsub231bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xba,0x52,0x80]
          vfmsub231bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfmsub231bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xba,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfmsub231bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfmsub231bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xba,0x94,0x87,0x23,0x01,0x00,0x00]
          vfmsub231bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfmsub231bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xba,0x10]
          vfmsub231bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfmsub231bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xba,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfmsub231bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfmsub231bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xba,0x51,0x7f]
          vfmsub231bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfmsub231bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xba,0x52,0x80]
          vfmsub231bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmadd132bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9c,0xd4]
          vfnmadd132bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfnmadd132bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9c,0xd4]
          vfnmadd132bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfnmadd132bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9c,0xd4]
          vfnmadd132bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmadd132bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9c,0xd4]
          vfnmadd132bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfnmadd132bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9c,0xd4]
          vfnmadd132bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfnmadd132bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9c,0xd4]
          vfnmadd132bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmadd132bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9c,0xd4]
          vfnmadd132bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfnmadd132bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9c,0xd4]
          vfnmadd132bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfnmadd132bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9c,0xd4]
          vfnmadd132bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmadd132bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfnmadd132bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9c,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd132bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfnmadd132bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x9c,0x10]
          vfnmadd132bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfnmadd132bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd132bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfnmadd132bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9c,0x51,0x7f]
          vfnmadd132bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmadd132bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x9c,0x52,0x80]
          vfnmadd132bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmadd132bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfnmadd132bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9c,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd132bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfnmadd132bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x9c,0x10]
          vfnmadd132bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfnmadd132bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd132bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfnmadd132bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9c,0x51,0x7f]
          vfnmadd132bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmadd132bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x9c,0x52,0x80]
          vfnmadd132bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmadd132bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd132bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfnmadd132bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9c,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd132bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfnmadd132bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x9c,0x10]
          vfnmadd132bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfnmadd132bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd132bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfnmadd132bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9c,0x51,0x7f]
          vfnmadd132bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmadd132bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x9c,0x52,0x80]
          vfnmadd132bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmadd213bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xac,0xd4]
          vfnmadd213bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfnmadd213bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xac,0xd4]
          vfnmadd213bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfnmadd213bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xac,0xd4]
          vfnmadd213bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmadd213bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xac,0xd4]
          vfnmadd213bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfnmadd213bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xac,0xd4]
          vfnmadd213bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfnmadd213bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xac,0xd4]
          vfnmadd213bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmadd213bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xac,0xd4]
          vfnmadd213bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfnmadd213bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xac,0xd4]
          vfnmadd213bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfnmadd213bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xac,0xd4]
          vfnmadd213bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmadd213bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xac,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfnmadd213bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xac,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd213bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfnmadd213bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xac,0x10]
          vfnmadd213bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfnmadd213bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xac,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd213bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfnmadd213bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xac,0x51,0x7f]
          vfnmadd213bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmadd213bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xac,0x52,0x80]
          vfnmadd213bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmadd213bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xac,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfnmadd213bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xac,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd213bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfnmadd213bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xac,0x10]
          vfnmadd213bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfnmadd213bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xac,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd213bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfnmadd213bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xac,0x51,0x7f]
          vfnmadd213bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmadd213bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xac,0x52,0x80]
          vfnmadd213bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmadd213bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xac,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd213bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfnmadd213bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xac,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd213bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfnmadd213bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xac,0x10]
          vfnmadd213bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfnmadd213bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xac,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd213bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfnmadd213bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xac,0x51,0x7f]
          vfnmadd213bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmadd213bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xac,0x52,0x80]
          vfnmadd213bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmadd231bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbc,0xd4]
          vfnmadd231bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfnmadd231bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbc,0xd4]
          vfnmadd231bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfnmadd231bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbc,0xd4]
          vfnmadd231bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmadd231bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbc,0xd4]
          vfnmadd231bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfnmadd231bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbc,0xd4]
          vfnmadd231bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfnmadd231bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbc,0xd4]
          vfnmadd231bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmadd231bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbc,0xd4]
          vfnmadd231bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfnmadd231bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbc,0xd4]
          vfnmadd231bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfnmadd231bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbc,0xd4]
          vfnmadd231bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmadd231bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbc,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfnmadd231bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbc,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd231bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfnmadd231bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xbc,0x10]
          vfnmadd231bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfnmadd231bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbc,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmadd231bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfnmadd231bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbc,0x51,0x7f]
          vfnmadd231bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmadd231bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xbc,0x52,0x80]
          vfnmadd231bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmadd231bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbc,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfnmadd231bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbc,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd231bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfnmadd231bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xbc,0x10]
          vfnmadd231bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfnmadd231bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbc,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmadd231bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfnmadd231bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbc,0x51,0x7f]
          vfnmadd231bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmadd231bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xbc,0x52,0x80]
          vfnmadd231bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmadd231bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbc,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmadd231bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfnmadd231bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbc,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmadd231bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfnmadd231bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xbc,0x10]
          vfnmadd231bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfnmadd231bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbc,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmadd231bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfnmadd231bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbc,0x51,0x7f]
          vfnmadd231bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmadd231bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xbc,0x52,0x80]
          vfnmadd231bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmsub132bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9e,0xd4]
          vfnmsub132bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfnmsub132bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9e,0xd4]
          vfnmsub132bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfnmsub132bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9e,0xd4]
          vfnmsub132bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmsub132bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9e,0xd4]
          vfnmsub132bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfnmsub132bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9e,0xd4]
          vfnmsub132bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfnmsub132bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9e,0xd4]
          vfnmsub132bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmsub132bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9e,0xd4]
          vfnmsub132bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfnmsub132bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9e,0xd4]
          vfnmsub132bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfnmsub132bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9e,0xd4]
          vfnmsub132bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmsub132bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfnmsub132bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x9e,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub132bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfnmsub132bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x9e,0x10]
          vfnmsub132bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfnmsub132bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x9e,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub132bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfnmsub132bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x9e,0x51,0x7f]
          vfnmsub132bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmsub132bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x9e,0x52,0x80]
          vfnmsub132bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmsub132bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfnmsub132bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x9e,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub132bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfnmsub132bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x9e,0x10]
          vfnmsub132bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfnmsub132bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x9e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub132bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfnmsub132bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x9e,0x51,0x7f]
          vfnmsub132bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmsub132bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x9e,0x52,0x80]
          vfnmsub132bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmsub132bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub132bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfnmsub132bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x9e,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub132bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfnmsub132bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x9e,0x10]
          vfnmsub132bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfnmsub132bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x9e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub132bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfnmsub132bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x9e,0x51,0x7f]
          vfnmsub132bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmsub132bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x9e,0x52,0x80]
          vfnmsub132bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmsub213bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xae,0xd4]
          vfnmsub213bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfnmsub213bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xae,0xd4]
          vfnmsub213bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfnmsub213bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xae,0xd4]
          vfnmsub213bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmsub213bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xae,0xd4]
          vfnmsub213bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfnmsub213bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xae,0xd4]
          vfnmsub213bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfnmsub213bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xae,0xd4]
          vfnmsub213bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmsub213bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xae,0xd4]
          vfnmsub213bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfnmsub213bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xae,0xd4]
          vfnmsub213bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfnmsub213bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xae,0xd4]
          vfnmsub213bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmsub213bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xae,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfnmsub213bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xae,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub213bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfnmsub213bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xae,0x10]
          vfnmsub213bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfnmsub213bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xae,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub213bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfnmsub213bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xae,0x51,0x7f]
          vfnmsub213bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmsub213bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xae,0x52,0x80]
          vfnmsub213bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmsub213bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xae,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfnmsub213bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xae,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub213bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfnmsub213bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xae,0x10]
          vfnmsub213bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfnmsub213bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xae,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub213bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfnmsub213bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xae,0x51,0x7f]
          vfnmsub213bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmsub213bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xae,0x52,0x80]
          vfnmsub213bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmsub213bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xae,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub213bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfnmsub213bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xae,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub213bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfnmsub213bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xae,0x10]
          vfnmsub213bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfnmsub213bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xae,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub213bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfnmsub213bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xae,0x51,0x7f]
          vfnmsub213bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmsub213bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xae,0x52,0x80]
          vfnmsub213bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmsub231bf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbe,0xd4]
          vfnmsub231bf16 %ymm4, %ymm3, %ymm2

// CHECK: vfnmsub231bf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbe,0xd4]
          vfnmsub231bf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vfnmsub231bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbe,0xd4]
          vfnmsub231bf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmsub231bf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbe,0xd4]
          vfnmsub231bf16 %zmm4, %zmm3, %zmm2

// CHECK: vfnmsub231bf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbe,0xd4]
          vfnmsub231bf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vfnmsub231bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbe,0xd4]
          vfnmsub231bf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmsub231bf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbe,0xd4]
          vfnmsub231bf16 %xmm4, %xmm3, %xmm2

// CHECK: vfnmsub231bf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbe,0xd4]
          vfnmsub231bf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vfnmsub231bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbe,0xd4]
          vfnmsub231bf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmsub231bf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbe,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231bf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vfnmsub231bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0xbe,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub231bf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vfnmsub231bf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0xbe,0x10]
          vfnmsub231bf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vfnmsub231bf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0xbe,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vfnmsub231bf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vfnmsub231bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0xbe,0x51,0x7f]
          vfnmsub231bf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmsub231bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0xbe,0x52,0x80]
          vfnmsub231bf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vfnmsub231bf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbe,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231bf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vfnmsub231bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0xbe,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub231bf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vfnmsub231bf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0xbe,0x10]
          vfnmsub231bf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vfnmsub231bf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0xbe,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vfnmsub231bf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vfnmsub231bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0xbe,0x51,0x7f]
          vfnmsub231bf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmsub231bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0xbe,0x52,0x80]
          vfnmsub231bf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vfnmsub231bf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbe,0x94,0xf4,0x00,0x00,0x00,0x10]
          vfnmsub231bf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vfnmsub231bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0xbe,0x94,0x87,0x23,0x01,0x00,0x00]
          vfnmsub231bf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vfnmsub231bf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0xbe,0x10]
          vfnmsub231bf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vfnmsub231bf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0xbe,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vfnmsub231bf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vfnmsub231bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0xbe,0x51,0x7f]
          vfnmsub231bf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vfnmsub231bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0xbe,0x52,0x80]
          vfnmsub231bf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vfpclassbf16 $123, %zmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0xeb,0x7b]
          vfpclassbf16 $123, %zmm3, %k5

// CHECK: vfpclassbf16 $123, %zmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0xeb,0x7b]
          vfpclassbf16 $123, %zmm3, %k5 {%k7}

// CHECK: vfpclassbf16 $123, %ymm3, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0xeb,0x7b]
          vfpclassbf16 $123, %ymm3, %k5

// CHECK: vfpclassbf16 $123, %ymm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0xeb,0x7b]
          vfpclassbf16 $123, %ymm3, %k5 {%k7}

// CHECK: vfpclassbf16 $123, %xmm3, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0xeb,0x7b]
          vfpclassbf16 $123, %xmm3, %k5

// CHECK: vfpclassbf16 $123, %xmm3, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0xeb,0x7b]
          vfpclassbf16 $123, %xmm3, %k5 {%k7}

// CHECK: vfpclassbf16x  $123, 268435456(%esp,%esi,8), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0xac,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vfpclassbf16x  $123, 268435456(%esp,%esi,8), %k5

// CHECK: vfpclassbf16x  $123, 291(%edi,%eax,4), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0xac,0x87,0x23,0x01,0x00,0x00,0x7b]
          vfpclassbf16x  $123, 291(%edi,%eax,4), %k5 {%k7}

// CHECK: vfpclassbf16  $123, (%eax){1to8}, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x66,0x28,0x7b]
          vfpclassbf16  $123, (%eax){1to8}, %k5

// CHECK: vfpclassbf16x  $123, -512(,%ebp,2), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x66,0x2c,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vfpclassbf16x  $123, -512(,%ebp,2), %k5

// CHECK: vfpclassbf16x  $123, 2032(%ecx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16x  $123, 2032(%ecx), %k5 {%k7}

// CHECK: vfpclassbf16  $123, -256(%edx){1to8}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x1f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16  $123, -256(%edx){1to8}, %k5 {%k7}

// CHECK: vfpclassbf16  $123, (%eax){1to16}, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x66,0x28,0x7b]
          vfpclassbf16  $123, (%eax){1to16}, %k5

// CHECK: vfpclassbf16y  $123, -1024(,%ebp,2), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x66,0x2c,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vfpclassbf16y  $123, -1024(,%ebp,2), %k5

// CHECK: vfpclassbf16y  $123, 4064(%ecx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16y  $123, 4064(%ecx), %k5 {%k7}

// CHECK: vfpclassbf16  $123, -256(%edx){1to16}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x3f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16  $123, -256(%edx){1to16}, %k5 {%k7}

// CHECK: vfpclassbf16  $123, (%eax){1to32}, %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x66,0x28,0x7b]
          vfpclassbf16  $123, (%eax){1to32}, %k5

// CHECK: vfpclassbf16z  $123, -2048(,%ebp,2), %k5
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x66,0x2c,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vfpclassbf16z  $123, -2048(,%ebp,2), %k5

// CHECK: vfpclassbf16z  $123, 8128(%ecx), %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x66,0x69,0x7f,0x7b]
          vfpclassbf16z  $123, 8128(%ecx), %k5 {%k7}

// CHECK: vfpclassbf16  $123, -256(%edx){1to32}, %k5 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x5f,0x66,0x6a,0x80,0x7b]
          vfpclassbf16  $123, -256(%edx){1to32}, %k5 {%k7}

// CHECK: vgetexpbf16 %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x42,0xd3]
          vgetexpbf16 %xmm3, %xmm2

// CHECK: vgetexpbf16 %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x42,0xd3]
          vgetexpbf16 %xmm3, %xmm2 {%k7}

// CHECK: vgetexpbf16 %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x42,0xd3]
          vgetexpbf16 %xmm3, %xmm2 {%k7} {z}

// CHECK: vgetexpbf16 %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x42,0xd3]
          vgetexpbf16 %zmm3, %zmm2

// CHECK: vgetexpbf16 %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x42,0xd3]
          vgetexpbf16 %zmm3, %zmm2 {%k7}

// CHECK: vgetexpbf16 %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x42,0xd3]
          vgetexpbf16 %zmm3, %zmm2 {%k7} {z}

// CHECK: vgetexpbf16 %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x42,0xd3]
          vgetexpbf16 %ymm3, %ymm2

// CHECK: vgetexpbf16 %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x42,0xd3]
          vgetexpbf16 %ymm3, %ymm2 {%k7}

// CHECK: vgetexpbf16 %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x42,0xd3]
          vgetexpbf16 %ymm3, %ymm2 {%k7} {z}

// CHECK: vgetexpbf16  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x42,0x94,0xf4,0x00,0x00,0x00,0x10]
          vgetexpbf16  268435456(%esp,%esi,8), %xmm2

// CHECK: vgetexpbf16  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x42,0x94,0x87,0x23,0x01,0x00,0x00]
          vgetexpbf16  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vgetexpbf16  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x42,0x10]
          vgetexpbf16  (%eax){1to8}, %xmm2

// CHECK: vgetexpbf16  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x42,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vgetexpbf16  -512(,%ebp,2), %xmm2

// CHECK: vgetexpbf16  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x42,0x51,0x7f]
          vgetexpbf16  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vgetexpbf16  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x42,0x52,0x80]
          vgetexpbf16  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vgetexpbf16  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x42,0x94,0xf4,0x00,0x00,0x00,0x10]
          vgetexpbf16  268435456(%esp,%esi,8), %ymm2

// CHECK: vgetexpbf16  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x42,0x94,0x87,0x23,0x01,0x00,0x00]
          vgetexpbf16  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vgetexpbf16  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x42,0x10]
          vgetexpbf16  (%eax){1to16}, %ymm2

// CHECK: vgetexpbf16  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x42,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vgetexpbf16  -1024(,%ebp,2), %ymm2

// CHECK: vgetexpbf16  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x42,0x51,0x7f]
          vgetexpbf16  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vgetexpbf16  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x42,0x52,0x80]
          vgetexpbf16  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vgetexpbf16  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x42,0x94,0xf4,0x00,0x00,0x00,0x10]
          vgetexpbf16  268435456(%esp,%esi,8), %zmm2

// CHECK: vgetexpbf16  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x42,0x94,0x87,0x23,0x01,0x00,0x00]
          vgetexpbf16  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vgetexpbf16  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x42,0x10]
          vgetexpbf16  (%eax){1to32}, %zmm2

// CHECK: vgetexpbf16  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x42,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vgetexpbf16  -2048(,%ebp,2), %zmm2

// CHECK: vgetexpbf16  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x42,0x51,0x7f]
          vgetexpbf16  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vgetexpbf16  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x42,0x52,0x80]
          vgetexpbf16  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vgetmantbf16 $123, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x26,0xd3,0x7b]
          vgetmantbf16 $123, %zmm3, %zmm2

// CHECK: vgetmantbf16 $123, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x26,0xd3,0x7b]
          vgetmantbf16 $123, %zmm3, %zmm2 {%k7}

// CHECK: vgetmantbf16 $123, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x26,0xd3,0x7b]
          vgetmantbf16 $123, %zmm3, %zmm2 {%k7} {z}

// CHECK: vgetmantbf16 $123, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x26,0xd3,0x7b]
          vgetmantbf16 $123, %ymm3, %ymm2

// CHECK: vgetmantbf16 $123, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x26,0xd3,0x7b]
          vgetmantbf16 $123, %ymm3, %ymm2 {%k7}

// CHECK: vgetmantbf16 $123, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x26,0xd3,0x7b]
          vgetmantbf16 $123, %ymm3, %ymm2 {%k7} {z}

// CHECK: vgetmantbf16 $123, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x26,0xd3,0x7b]
          vgetmantbf16 $123, %xmm3, %xmm2

// CHECK: vgetmantbf16 $123, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x26,0xd3,0x7b]
          vgetmantbf16 $123, %xmm3, %xmm2 {%k7}

// CHECK: vgetmantbf16 $123, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x26,0xd3,0x7b]
          vgetmantbf16 $123, %xmm3, %xmm2 {%k7} {z}

// CHECK: vgetmantbf16  $123, 268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x26,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16  $123, 268435456(%esp,%esi,8), %xmm2

// CHECK: vgetmantbf16  $123, 291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x26,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16  $123, 291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vgetmantbf16  $123, (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x26,0x10,0x7b]
          vgetmantbf16  $123, (%eax){1to8}, %xmm2

// CHECK: vgetmantbf16  $123, -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x26,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vgetmantbf16  $123, -512(,%ebp,2), %xmm2

// CHECK: vgetmantbf16  $123, 2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x26,0x51,0x7f,0x7b]
          vgetmantbf16  $123, 2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vgetmantbf16  $123, -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0x9f,0x26,0x52,0x80,0x7b]
          vgetmantbf16  $123, -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vgetmantbf16  $123, 268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x26,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16  $123, 268435456(%esp,%esi,8), %ymm2

// CHECK: vgetmantbf16  $123, 291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x26,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16  $123, 291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vgetmantbf16  $123, (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x26,0x10,0x7b]
          vgetmantbf16  $123, (%eax){1to16}, %ymm2

// CHECK: vgetmantbf16  $123, -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x26,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vgetmantbf16  $123, -1024(,%ebp,2), %ymm2

// CHECK: vgetmantbf16  $123, 4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x26,0x51,0x7f,0x7b]
          vgetmantbf16  $123, 4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vgetmantbf16  $123, -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xbf,0x26,0x52,0x80,0x7b]
          vgetmantbf16  $123, -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vgetmantbf16  $123, 268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x26,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vgetmantbf16  $123, 268435456(%esp,%esi,8), %zmm2

// CHECK: vgetmantbf16  $123, 291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x26,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vgetmantbf16  $123, 291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vgetmantbf16  $123, (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x26,0x10,0x7b]
          vgetmantbf16  $123, (%eax){1to32}, %zmm2

// CHECK: vgetmantbf16  $123, -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x26,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vgetmantbf16  $123, -2048(,%ebp,2), %zmm2

// CHECK: vgetmantbf16  $123, 8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x26,0x51,0x7f,0x7b]
          vgetmantbf16  $123, 8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vgetmantbf16  $123, -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xdf,0x26,0x52,0x80,0x7b]
          vgetmantbf16  $123, -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vmaxbf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5f,0xd4]
          vmaxbf16 %ymm4, %ymm3, %ymm2

// CHECK: vmaxbf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5f,0xd4]
          vmaxbf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vmaxbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5f,0xd4]
          vmaxbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vmaxbf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5f,0xd4]
          vmaxbf16 %zmm4, %zmm3, %zmm2

// CHECK: vmaxbf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5f,0xd4]
          vmaxbf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vmaxbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5f,0xd4]
          vmaxbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vmaxbf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5f,0xd4]
          vmaxbf16 %xmm4, %xmm3, %xmm2

// CHECK: vmaxbf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5f,0xd4]
          vmaxbf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vmaxbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5f,0xd4]
          vmaxbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vmaxbf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmaxbf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vmaxbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5f,0x94,0x87,0x23,0x01,0x00,0x00]
          vmaxbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vmaxbf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5f,0x10]
          vmaxbf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vmaxbf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5f,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vmaxbf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vmaxbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5f,0x51,0x7f]
          vmaxbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vmaxbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5f,0x52,0x80]
          vmaxbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vmaxbf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmaxbf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vmaxbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5f,0x94,0x87,0x23,0x01,0x00,0x00]
          vmaxbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vmaxbf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5f,0x10]
          vmaxbf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vmaxbf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5f,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vmaxbf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vmaxbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5f,0x51,0x7f]
          vmaxbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vmaxbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5f,0x52,0x80]
          vmaxbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vmaxbf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5f,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmaxbf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vmaxbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5f,0x94,0x87,0x23,0x01,0x00,0x00]
          vmaxbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vmaxbf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5f,0x10]
          vmaxbf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vmaxbf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5f,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vmaxbf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vmaxbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5f,0x51,0x7f]
          vmaxbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vmaxbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5f,0x52,0x80]
          vmaxbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminbf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5d,0xd4]
          vminbf16 %ymm4, %ymm3, %ymm2

// CHECK: vminbf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5d,0xd4]
          vminbf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vminbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5d,0xd4]
          vminbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminbf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5d,0xd4]
          vminbf16 %zmm4, %zmm3, %zmm2

// CHECK: vminbf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5d,0xd4]
          vminbf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vminbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5d,0xd4]
          vminbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminbf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5d,0xd4]
          vminbf16 %xmm4, %xmm3, %xmm2

// CHECK: vminbf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5d,0xd4]
          vminbf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vminbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5d,0xd4]
          vminbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminbf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vminbf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vminbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5d,0x94,0x87,0x23,0x01,0x00,0x00]
          vminbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vminbf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5d,0x10]
          vminbf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vminbf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5d,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vminbf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vminbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5d,0x51,0x7f]
          vminbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vminbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5d,0x52,0x80]
          vminbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminbf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vminbf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vminbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5d,0x94,0x87,0x23,0x01,0x00,0x00]
          vminbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vminbf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5d,0x10]
          vminbf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vminbf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5d,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vminbf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vminbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5d,0x51,0x7f]
          vminbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vminbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5d,0x52,0x80]
          vminbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminbf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5d,0x94,0xf4,0x00,0x00,0x00,0x10]
          vminbf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vminbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5d,0x94,0x87,0x23,0x01,0x00,0x00]
          vminbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vminbf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5d,0x10]
          vminbf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vminbf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5d,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vminbf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vminbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5d,0x51,0x7f]
          vminbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5d,0x52,0x80]
          vminbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vmulbf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x59,0xd4]
          vmulbf16 %ymm4, %ymm3, %ymm2

// CHECK: vmulbf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x59,0xd4]
          vmulbf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vmulbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x59,0xd4]
          vmulbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vmulbf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x59,0xd4]
          vmulbf16 %zmm4, %zmm3, %zmm2

// CHECK: vmulbf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x59,0xd4]
          vmulbf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vmulbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x59,0xd4]
          vmulbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vmulbf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x59,0xd4]
          vmulbf16 %xmm4, %xmm3, %xmm2

// CHECK: vmulbf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x59,0xd4]
          vmulbf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vmulbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x59,0xd4]
          vmulbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vmulbf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x59,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmulbf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vmulbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x59,0x94,0x87,0x23,0x01,0x00,0x00]
          vmulbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vmulbf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x59,0x10]
          vmulbf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vmulbf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x59,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vmulbf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vmulbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x59,0x51,0x7f]
          vmulbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vmulbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x59,0x52,0x80]
          vmulbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vmulbf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x59,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmulbf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vmulbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x59,0x94,0x87,0x23,0x01,0x00,0x00]
          vmulbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vmulbf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x59,0x10]
          vmulbf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vmulbf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x59,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vmulbf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vmulbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x59,0x51,0x7f]
          vmulbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vmulbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x59,0x52,0x80]
          vmulbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vmulbf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x59,0x94,0xf4,0x00,0x00,0x00,0x10]
          vmulbf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vmulbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x59,0x94,0x87,0x23,0x01,0x00,0x00]
          vmulbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vmulbf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x59,0x10]
          vmulbf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vmulbf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x59,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vmulbf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vmulbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x59,0x51,0x7f]
          vmulbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vmulbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x59,0x52,0x80]
          vmulbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vrcpbf16 %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4c,0xd3]
          vrcpbf16 %xmm3, %xmm2

// CHECK: vrcpbf16 %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4c,0xd3]
          vrcpbf16 %xmm3, %xmm2 {%k7}

// CHECK: vrcpbf16 %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4c,0xd3]
          vrcpbf16 %xmm3, %xmm2 {%k7} {z}

// CHECK: vrcpbf16 %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4c,0xd3]
          vrcpbf16 %zmm3, %zmm2

// CHECK: vrcpbf16 %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4c,0xd3]
          vrcpbf16 %zmm3, %zmm2 {%k7}

// CHECK: vrcpbf16 %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4c,0xd3]
          vrcpbf16 %zmm3, %zmm2 {%k7} {z}

// CHECK: vrcpbf16 %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4c,0xd3]
          vrcpbf16 %ymm3, %ymm2

// CHECK: vrcpbf16 %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4c,0xd3]
          vrcpbf16 %ymm3, %ymm2 {%k7}

// CHECK: vrcpbf16 %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4c,0xd3]
          vrcpbf16 %ymm3, %ymm2 {%k7} {z}

// CHECK: vrcpbf16  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrcpbf16  268435456(%esp,%esi,8), %xmm2

// CHECK: vrcpbf16  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4c,0x94,0x87,0x23,0x01,0x00,0x00]
          vrcpbf16  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vrcpbf16  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x18,0x4c,0x10]
          vrcpbf16  (%eax){1to8}, %xmm2

// CHECK: vrcpbf16  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vrcpbf16  -512(,%ebp,2), %xmm2

// CHECK: vrcpbf16  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4c,0x51,0x7f]
          vrcpbf16  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vrcpbf16  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0x9f,0x4c,0x52,0x80]
          vrcpbf16  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vrcpbf16  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrcpbf16  268435456(%esp,%esi,8), %ymm2

// CHECK: vrcpbf16  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4c,0x94,0x87,0x23,0x01,0x00,0x00]
          vrcpbf16  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vrcpbf16  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x38,0x4c,0x10]
          vrcpbf16  (%eax){1to16}, %ymm2

// CHECK: vrcpbf16  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vrcpbf16  -1024(,%ebp,2), %ymm2

// CHECK: vrcpbf16  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4c,0x51,0x7f]
          vrcpbf16  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vrcpbf16  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xbf,0x4c,0x52,0x80]
          vrcpbf16  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vrcpbf16  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrcpbf16  268435456(%esp,%esi,8), %zmm2

// CHECK: vrcpbf16  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4c,0x94,0x87,0x23,0x01,0x00,0x00]
          vrcpbf16  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vrcpbf16  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x58,0x4c,0x10]
          vrcpbf16  (%eax){1to32}, %zmm2

// CHECK: vrcpbf16  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vrcpbf16  -2048(,%ebp,2), %zmm2

// CHECK: vrcpbf16  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4c,0x51,0x7f]
          vrcpbf16  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vrcpbf16  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xdf,0x4c,0x52,0x80]
          vrcpbf16  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vreducebf16 $123, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x56,0xd3,0x7b]
          vreducebf16 $123, %zmm3, %zmm2

// CHECK: vreducebf16 $123, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x56,0xd3,0x7b]
          vreducebf16 $123, %zmm3, %zmm2 {%k7}

// CHECK: vreducebf16 $123, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x56,0xd3,0x7b]
          vreducebf16 $123, %zmm3, %zmm2 {%k7} {z}

// CHECK: vreducebf16 $123, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x56,0xd3,0x7b]
          vreducebf16 $123, %ymm3, %ymm2

// CHECK: vreducebf16 $123, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x56,0xd3,0x7b]
          vreducebf16 $123, %ymm3, %ymm2 {%k7}

// CHECK: vreducebf16 $123, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x56,0xd3,0x7b]
          vreducebf16 $123, %ymm3, %ymm2 {%k7} {z}

// CHECK: vreducebf16 $123, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x56,0xd3,0x7b]
          vreducebf16 $123, %xmm3, %xmm2

// CHECK: vreducebf16 $123, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x56,0xd3,0x7b]
          vreducebf16 $123, %xmm3, %xmm2 {%k7}

// CHECK: vreducebf16 $123, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x56,0xd3,0x7b]
          vreducebf16 $123, %xmm3, %xmm2 {%k7} {z}

// CHECK: vreducebf16  $123, 268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x56,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16  $123, 268435456(%esp,%esi,8), %xmm2

// CHECK: vreducebf16  $123, 291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x56,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16  $123, 291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vreducebf16  $123, (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x56,0x10,0x7b]
          vreducebf16  $123, (%eax){1to8}, %xmm2

// CHECK: vreducebf16  $123, -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x56,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vreducebf16  $123, -512(,%ebp,2), %xmm2

// CHECK: vreducebf16  $123, 2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x56,0x51,0x7f,0x7b]
          vreducebf16  $123, 2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vreducebf16  $123, -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0x9f,0x56,0x52,0x80,0x7b]
          vreducebf16  $123, -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vreducebf16  $123, 268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x56,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16  $123, 268435456(%esp,%esi,8), %ymm2

// CHECK: vreducebf16  $123, 291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x56,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16  $123, 291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vreducebf16  $123, (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x56,0x10,0x7b]
          vreducebf16  $123, (%eax){1to16}, %ymm2

// CHECK: vreducebf16  $123, -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x56,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vreducebf16  $123, -1024(,%ebp,2), %ymm2

// CHECK: vreducebf16  $123, 4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x56,0x51,0x7f,0x7b]
          vreducebf16  $123, 4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vreducebf16  $123, -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xbf,0x56,0x52,0x80,0x7b]
          vreducebf16  $123, -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vreducebf16  $123, 268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x56,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vreducebf16  $123, 268435456(%esp,%esi,8), %zmm2

// CHECK: vreducebf16  $123, 291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x56,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vreducebf16  $123, 291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vreducebf16  $123, (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x56,0x10,0x7b]
          vreducebf16  $123, (%eax){1to32}, %zmm2

// CHECK: vreducebf16  $123, -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x56,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vreducebf16  $123, -2048(,%ebp,2), %zmm2

// CHECK: vreducebf16  $123, 8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x56,0x51,0x7f,0x7b]
          vreducebf16  $123, 8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vreducebf16  $123, -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xdf,0x56,0x52,0x80,0x7b]
          vreducebf16  $123, -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vrndscalebf16 $123, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x08,0xd3,0x7b]
          vrndscalebf16 $123, %zmm3, %zmm2

// CHECK: vrndscalebf16 $123, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x08,0xd3,0x7b]
          vrndscalebf16 $123, %zmm3, %zmm2 {%k7}

// CHECK: vrndscalebf16 $123, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x08,0xd3,0x7b]
          vrndscalebf16 $123, %zmm3, %zmm2 {%k7} {z}

// CHECK: vrndscalebf16 $123, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x08,0xd3,0x7b]
          vrndscalebf16 $123, %ymm3, %ymm2

// CHECK: vrndscalebf16 $123, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x08,0xd3,0x7b]
          vrndscalebf16 $123, %ymm3, %ymm2 {%k7}

// CHECK: vrndscalebf16 $123, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x08,0xd3,0x7b]
          vrndscalebf16 $123, %ymm3, %ymm2 {%k7} {z}

// CHECK: vrndscalebf16 $123, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x08,0xd3,0x7b]
          vrndscalebf16 $123, %xmm3, %xmm2

// CHECK: vrndscalebf16 $123, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x08,0xd3,0x7b]
          vrndscalebf16 $123, %xmm3, %xmm2 {%k7}

// CHECK: vrndscalebf16 $123, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x08,0xd3,0x7b]
          vrndscalebf16 $123, %xmm3, %xmm2 {%k7} {z}

// CHECK: vrndscalebf16  $123, 268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x08,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16  $123, 268435456(%esp,%esi,8), %xmm2

// CHECK: vrndscalebf16  $123, 291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x0f,0x08,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16  $123, 291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vrndscalebf16  $123, (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x18,0x08,0x10,0x7b]
          vrndscalebf16  $123, (%eax){1to8}, %xmm2

// CHECK: vrndscalebf16  $123, -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x08,0x08,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vrndscalebf16  $123, -512(,%ebp,2), %xmm2

// CHECK: vrndscalebf16  $123, 2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0x8f,0x08,0x51,0x7f,0x7b]
          vrndscalebf16  $123, 2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vrndscalebf16  $123, -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0x9f,0x08,0x52,0x80,0x7b]
          vrndscalebf16  $123, -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vrndscalebf16  $123, 268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x08,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16  $123, 268435456(%esp,%esi,8), %ymm2

// CHECK: vrndscalebf16  $123, 291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x2f,0x08,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16  $123, 291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vrndscalebf16  $123, (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x38,0x08,0x10,0x7b]
          vrndscalebf16  $123, (%eax){1to16}, %ymm2

// CHECK: vrndscalebf16  $123, -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x28,0x08,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vrndscalebf16  $123, -1024(,%ebp,2), %ymm2

// CHECK: vrndscalebf16  $123, 4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xaf,0x08,0x51,0x7f,0x7b]
          vrndscalebf16  $123, 4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vrndscalebf16  $123, -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xbf,0x08,0x52,0x80,0x7b]
          vrndscalebf16  $123, -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vrndscalebf16  $123, 268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x08,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vrndscalebf16  $123, 268435456(%esp,%esi,8), %zmm2

// CHECK: vrndscalebf16  $123, 291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x7f,0x4f,0x08,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vrndscalebf16  $123, 291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vrndscalebf16  $123, (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x58,0x08,0x10,0x7b]
          vrndscalebf16  $123, (%eax){1to32}, %zmm2

// CHECK: vrndscalebf16  $123, -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf3,0x7f,0x48,0x08,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vrndscalebf16  $123, -2048(,%ebp,2), %zmm2

// CHECK: vrndscalebf16  $123, 8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xcf,0x08,0x51,0x7f,0x7b]
          vrndscalebf16  $123, 8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vrndscalebf16  $123, -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x7f,0xdf,0x08,0x52,0x80,0x7b]
          vrndscalebf16  $123, -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vrsqrtbf16 %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4e,0xd3]
          vrsqrtbf16 %xmm3, %xmm2

// CHECK: vrsqrtbf16 %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4e,0xd3]
          vrsqrtbf16 %xmm3, %xmm2 {%k7}

// CHECK: vrsqrtbf16 %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4e,0xd3]
          vrsqrtbf16 %xmm3, %xmm2 {%k7} {z}

// CHECK: vrsqrtbf16 %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4e,0xd3]
          vrsqrtbf16 %zmm3, %zmm2

// CHECK: vrsqrtbf16 %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4e,0xd3]
          vrsqrtbf16 %zmm3, %zmm2 {%k7}

// CHECK: vrsqrtbf16 %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4e,0xd3]
          vrsqrtbf16 %zmm3, %zmm2 {%k7} {z}

// CHECK: vrsqrtbf16 %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4e,0xd3]
          vrsqrtbf16 %ymm3, %ymm2

// CHECK: vrsqrtbf16 %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4e,0xd3]
          vrsqrtbf16 %ymm3, %ymm2 {%k7}

// CHECK: vrsqrtbf16 %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4e,0xd3]
          vrsqrtbf16 %ymm3, %ymm2 {%k7} {z}

// CHECK: vrsqrtbf16  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtbf16  268435456(%esp,%esi,8), %xmm2

// CHECK: vrsqrtbf16  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x0f,0x4e,0x94,0x87,0x23,0x01,0x00,0x00]
          vrsqrtbf16  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vrsqrtbf16  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x18,0x4e,0x10]
          vrsqrtbf16  (%eax){1to8}, %xmm2

// CHECK: vrsqrtbf16  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x4e,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vrsqrtbf16  -512(,%ebp,2), %xmm2

// CHECK: vrsqrtbf16  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0x8f,0x4e,0x51,0x7f]
          vrsqrtbf16  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vrsqrtbf16  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0x9f,0x4e,0x52,0x80]
          vrsqrtbf16  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vrsqrtbf16  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtbf16  268435456(%esp,%esi,8), %ymm2

// CHECK: vrsqrtbf16  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x2f,0x4e,0x94,0x87,0x23,0x01,0x00,0x00]
          vrsqrtbf16  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vrsqrtbf16  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x38,0x4e,0x10]
          vrsqrtbf16  (%eax){1to16}, %ymm2

// CHECK: vrsqrtbf16  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x4e,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vrsqrtbf16  -1024(,%ebp,2), %ymm2

// CHECK: vrsqrtbf16  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xaf,0x4e,0x51,0x7f]
          vrsqrtbf16  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vrsqrtbf16  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xbf,0x4e,0x52,0x80]
          vrsqrtbf16  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vrsqrtbf16  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4e,0x94,0xf4,0x00,0x00,0x00,0x10]
          vrsqrtbf16  268435456(%esp,%esi,8), %zmm2

// CHECK: vrsqrtbf16  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x4e,0x94,0x87,0x23,0x01,0x00,0x00]
          vrsqrtbf16  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vrsqrtbf16  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x58,0x4e,0x10]
          vrsqrtbf16  (%eax){1to32}, %zmm2

// CHECK: vrsqrtbf16  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x4e,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vrsqrtbf16  -2048(,%ebp,2), %zmm2

// CHECK: vrsqrtbf16  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x4e,0x51,0x7f]
          vrsqrtbf16  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vrsqrtbf16  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x7c,0xdf,0x4e,0x52,0x80]
          vrsqrtbf16  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vscalefbf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x2c,0xd4]
          vscalefbf16 %ymm4, %ymm3, %ymm2

// CHECK: vscalefbf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x2c,0xd4]
          vscalefbf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vscalefbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x2c,0xd4]
          vscalefbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vscalefbf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x2c,0xd4]
          vscalefbf16 %zmm4, %zmm3, %zmm2

// CHECK: vscalefbf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x2c,0xd4]
          vscalefbf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vscalefbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x2c,0xd4]
          vscalefbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vscalefbf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x2c,0xd4]
          vscalefbf16 %xmm4, %xmm3, %xmm2

// CHECK: vscalefbf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x2c,0xd4]
          vscalefbf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vscalefbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x2c,0xd4]
          vscalefbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vscalefbf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vscalefbf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vscalefbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x4f,0x2c,0x94,0x87,0x23,0x01,0x00,0x00]
          vscalefbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vscalefbf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x58,0x2c,0x10]
          vscalefbf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vscalefbf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x48,0x2c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vscalefbf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vscalefbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xcf,0x2c,0x51,0x7f]
          vscalefbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vscalefbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xdf,0x2c,0x52,0x80]
          vscalefbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vscalefbf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vscalefbf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vscalefbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x2f,0x2c,0x94,0x87,0x23,0x01,0x00,0x00]
          vscalefbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vscalefbf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x38,0x2c,0x10]
          vscalefbf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vscalefbf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf6,0x64,0x28,0x2c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vscalefbf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vscalefbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xaf,0x2c,0x51,0x7f]
          vscalefbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vscalefbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0xbf,0x2c,0x52,0x80]
          vscalefbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vscalefbf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x2c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vscalefbf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vscalefbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf6,0x64,0x0f,0x2c,0x94,0x87,0x23,0x01,0x00,0x00]
          vscalefbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vscalefbf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x18,0x2c,0x10]
          vscalefbf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vscalefbf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf6,0x64,0x08,0x2c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vscalefbf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vscalefbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x8f,0x2c,0x51,0x7f]
          vscalefbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vscalefbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf6,0x64,0x9f,0x2c,0x52,0x80]
          vscalefbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vsqrtbf16 %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x51,0xd3]
          vsqrtbf16 %xmm3, %xmm2

// CHECK: vsqrtbf16 %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x51,0xd3]
          vsqrtbf16 %xmm3, %xmm2 {%k7}

// CHECK: vsqrtbf16 %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x51,0xd3]
          vsqrtbf16 %xmm3, %xmm2 {%k7} {z}

// CHECK: vsqrtbf16 %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x51,0xd3]
          vsqrtbf16 %zmm3, %zmm2

// CHECK: vsqrtbf16 %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x51,0xd3]
          vsqrtbf16 %zmm3, %zmm2 {%k7}

// CHECK: vsqrtbf16 %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x51,0xd3]
          vsqrtbf16 %zmm3, %zmm2 {%k7} {z}

// CHECK: vsqrtbf16 %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x51,0xd3]
          vsqrtbf16 %ymm3, %ymm2

// CHECK: vsqrtbf16 %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x51,0xd3]
          vsqrtbf16 %ymm3, %ymm2 {%k7}

// CHECK: vsqrtbf16 %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x51,0xd3]
          vsqrtbf16 %ymm3, %ymm2 {%k7} {z}

// CHECK: vsqrtbf16  268435456(%esp,%esi,8), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsqrtbf16  268435456(%esp,%esi,8), %xmm2

// CHECK: vsqrtbf16  291(%edi,%eax,4), %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x0f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vsqrtbf16  291(%edi,%eax,4), %xmm2 {%k7}

// CHECK: vsqrtbf16  (%eax){1to8}, %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x18,0x51,0x10]
          vsqrtbf16  (%eax){1to8}, %xmm2

// CHECK: vsqrtbf16  -512(,%ebp,2), %xmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x08,0x51,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vsqrtbf16  -512(,%ebp,2), %xmm2

// CHECK: vsqrtbf16  2032(%ecx), %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x8f,0x51,0x51,0x7f]
          vsqrtbf16  2032(%ecx), %xmm2 {%k7} {z}

// CHECK: vsqrtbf16  -256(%edx){1to8}, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0x9f,0x51,0x52,0x80]
          vsqrtbf16  -256(%edx){1to8}, %xmm2 {%k7} {z}

// CHECK: vsqrtbf16  268435456(%esp,%esi,8), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsqrtbf16  268435456(%esp,%esi,8), %ymm2

// CHECK: vsqrtbf16  291(%edi,%eax,4), %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x2f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vsqrtbf16  291(%edi,%eax,4), %ymm2 {%k7}

// CHECK: vsqrtbf16  (%eax){1to16}, %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x38,0x51,0x10]
          vsqrtbf16  (%eax){1to16}, %ymm2

// CHECK: vsqrtbf16  -1024(,%ebp,2), %ymm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x28,0x51,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vsqrtbf16  -1024(,%ebp,2), %ymm2

// CHECK: vsqrtbf16  4064(%ecx), %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xaf,0x51,0x51,0x7f]
          vsqrtbf16  4064(%ecx), %ymm2 {%k7} {z}

// CHECK: vsqrtbf16  -256(%edx){1to16}, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xbf,0x51,0x52,0x80]
          vsqrtbf16  -256(%edx){1to16}, %ymm2 {%k7} {z}

// CHECK: vsqrtbf16  268435456(%esp,%esi,8), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x51,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsqrtbf16  268435456(%esp,%esi,8), %zmm2

// CHECK: vsqrtbf16  291(%edi,%eax,4), %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x7d,0x4f,0x51,0x94,0x87,0x23,0x01,0x00,0x00]
          vsqrtbf16  291(%edi,%eax,4), %zmm2 {%k7}

// CHECK: vsqrtbf16  (%eax){1to32}, %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x58,0x51,0x10]
          vsqrtbf16  (%eax){1to32}, %zmm2

// CHECK: vsqrtbf16  -2048(,%ebp,2), %zmm2
// CHECK: encoding: [0x62,0xf5,0x7d,0x48,0x51,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vsqrtbf16  -2048(,%ebp,2), %zmm2

// CHECK: vsqrtbf16  8128(%ecx), %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xcf,0x51,0x51,0x7f]
          vsqrtbf16  8128(%ecx), %zmm2 {%k7} {z}

// CHECK: vsqrtbf16  -256(%edx){1to32}, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x7d,0xdf,0x51,0x52,0x80]
          vsqrtbf16  -256(%edx){1to32}, %zmm2 {%k7} {z}

// CHECK: vsubbf16 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5c,0xd4]
          vsubbf16 %ymm4, %ymm3, %ymm2

// CHECK: vsubbf16 %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5c,0xd4]
          vsubbf16 %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vsubbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5c,0xd4]
          vsubbf16 %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vsubbf16 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5c,0xd4]
          vsubbf16 %zmm4, %zmm3, %zmm2

// CHECK: vsubbf16 %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5c,0xd4]
          vsubbf16 %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vsubbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5c,0xd4]
          vsubbf16 %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vsubbf16 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5c,0xd4]
          vsubbf16 %xmm4, %xmm3, %xmm2

// CHECK: vsubbf16 %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5c,0xd4]
          vsubbf16 %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vsubbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5c,0xd4]
          vsubbf16 %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vsubbf16  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsubbf16  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vsubbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x4f,0x5c,0x94,0x87,0x23,0x01,0x00,0x00]
          vsubbf16  291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vsubbf16  (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x58,0x5c,0x10]
          vsubbf16  (%eax){1to32}, %zmm3, %zmm2

// CHECK: vsubbf16  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x48,0x5c,0x14,0x6d,0x00,0xf8,0xff,0xff]
          vsubbf16  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vsubbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xcf,0x5c,0x51,0x7f]
          vsubbf16  8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vsubbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xdf,0x5c,0x52,0x80]
          vsubbf16  -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vsubbf16  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsubbf16  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vsubbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x2f,0x5c,0x94,0x87,0x23,0x01,0x00,0x00]
          vsubbf16  291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vsubbf16  (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x38,0x5c,0x10]
          vsubbf16  (%eax){1to16}, %ymm3, %ymm2

// CHECK: vsubbf16  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf5,0x65,0x28,0x5c,0x14,0x6d,0x00,0xfc,0xff,0xff]
          vsubbf16  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vsubbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xaf,0x5c,0x51,0x7f]
          vsubbf16  4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vsubbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0xbf,0x5c,0x52,0x80]
          vsubbf16  -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vsubbf16  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5c,0x94,0xf4,0x00,0x00,0x00,0x10]
          vsubbf16  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vsubbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf5,0x65,0x0f,0x5c,0x94,0x87,0x23,0x01,0x00,0x00]
          vsubbf16  291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vsubbf16  (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x18,0x5c,0x10]
          vsubbf16  (%eax){1to8}, %xmm3, %xmm2

// CHECK: vsubbf16  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf5,0x65,0x08,0x5c,0x14,0x6d,0x00,0xfe,0xff,0xff]
          vsubbf16  -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vsubbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x8f,0x5c,0x51,0x7f]
          vsubbf16  2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vsubbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf5,0x65,0x9f,0x5c,0x52,0x80]
          vsubbf16  -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

