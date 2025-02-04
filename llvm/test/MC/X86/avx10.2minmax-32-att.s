// RUN: llvm-mc -triple i386 --show-encoding %s | FileCheck %s

// CHECK: vminmaxbf16 $123, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0x52,0xd4,0x7b]
          vminmaxbf16 $123, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxbf16 $123, %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0x52,0xd4,0x7b]
          vminmaxbf16 $123, %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vminmaxbf16 $123, %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x67,0x8f,0x52,0xd4,0x7b]
          vminmaxbf16 $123, %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxbf16 $123, %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0x52,0xd4,0x7b]
          vminmaxbf16 $123, %zmm4, %zmm3, %zmm2

// CHECK: vminmaxbf16 $123, %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0x52,0xd4,0x7b]
          vminmaxbf16 $123, %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vminmaxbf16 $123, %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x67,0xcf,0x52,0xd4,0x7b]
          vminmaxbf16 $123, %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxbf16 $123, %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0x52,0xd4,0x7b]
          vminmaxbf16 $123, %ymm4, %ymm3, %ymm2

// CHECK: vminmaxbf16 $123, %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0x52,0xd4,0x7b]
          vminmaxbf16 $123, %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vminmaxbf16 $123, %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x67,0xaf,0x52,0xd4,0x7b]
          vminmaxbf16 $123, %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxbf16  $123, 268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxbf16  $123, 268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vminmaxbf16  $123, 291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x2f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxbf16  $123, 291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vminmaxbf16  $123, (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x67,0x38,0x52,0x10,0x7b]
          vminmaxbf16  $123, (%eax){1to16}, %ymm3, %ymm2

// CHECK: vminmaxbf16  $123, -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x67,0x28,0x52,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxbf16  $123, -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vminmaxbf16  $123, 4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x67,0xaf,0x52,0x51,0x7f,0x7b]
          vminmaxbf16  $123, 4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxbf16  $123, -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x67,0xbf,0x52,0x52,0x80,0x7b]
          vminmaxbf16  $123, -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxbf16  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxbf16  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vminmaxbf16  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x0f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxbf16  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vminmaxbf16  $123, (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x67,0x18,0x52,0x10,0x7b]
          vminmaxbf16  $123, (%eax){1to8}, %xmm3, %xmm2

// CHECK: vminmaxbf16  $123, -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x67,0x08,0x52,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxbf16  $123, -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vminmaxbf16  $123, 2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x67,0x8f,0x52,0x51,0x7f,0x7b]
          vminmaxbf16  $123, 2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxbf16  $123, -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x67,0x9f,0x52,0x52,0x80,0x7b]
          vminmaxbf16  $123, -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxbf16  $123, 268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxbf16  $123, 268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vminmaxbf16  $123, 291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x67,0x4f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxbf16  $123, 291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vminmaxbf16  $123, (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x67,0x58,0x52,0x10,0x7b]
          vminmaxbf16  $123, (%eax){1to32}, %zmm3, %zmm2

// CHECK: vminmaxbf16  $123, -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x67,0x48,0x52,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxbf16  $123, -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vminmaxbf16  $123, 8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x67,0xcf,0x52,0x51,0x7f,0x7b]
          vminmaxbf16  $123, 8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxbf16  $123, -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x67,0xdf,0x52,0x52,0x80,0x7b]
          vminmaxbf16  $123, -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxpd $123, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x52,0xd4,0x7b]
          vminmaxpd $123, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxpd $123, %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0xe5,0x0f,0x52,0xd4,0x7b]
          vminmaxpd $123, %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vminmaxpd $123, %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0x8f,0x52,0xd4,0x7b]
          vminmaxpd $123, %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxpd $123, %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x48,0x52,0xd4,0x7b]
          vminmaxpd $123, %zmm4, %zmm3, %zmm2

// CHECK: vminmaxpd $123, {sae}, %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x18,0x52,0xd4,0x7b]
          vminmaxpd $123, {sae}, %zmm4, %zmm3, %zmm2

// CHECK: vminmaxpd $123, %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0xe5,0x4f,0x52,0xd4,0x7b]
          vminmaxpd $123, %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vminmaxpd $123, {sae}, %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0x9f,0x52,0xd4,0x7b]
          vminmaxpd $123, {sae}, %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxpd $123, %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x28,0x52,0xd4,0x7b]
          vminmaxpd $123, %ymm4, %ymm3, %ymm2

// CHECK: vminmaxpd $123, {sae}, %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0xe1,0x18,0x52,0xd4,0x7b]
          vminmaxpd $123, {sae}, %ymm4, %ymm3, %ymm2

// CHECK: vminmaxpd $123, %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0xe5,0x2f,0x52,0xd4,0x7b]
          vminmaxpd $123, %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vminmaxpd $123, {sae}, %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe1,0x9f,0x52,0xd4,0x7b]
          vminmaxpd $123, {sae}, %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxpd  $123, 268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x28,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd  $123, 268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vminmaxpd  $123, 291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0xe5,0x2f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd  $123, 291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vminmaxpd  $123, (%eax){1to4}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x38,0x52,0x10,0x7b]
          vminmaxpd  $123, (%eax){1to4}, %ymm3, %ymm2

// CHECK: vminmaxpd  $123, -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x28,0x52,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxpd  $123, -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vminmaxpd  $123, 4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0xaf,0x52,0x51,0x7f,0x7b]
          vminmaxpd  $123, 4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxpd  $123, -1024(%edx){1to4}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0xbf,0x52,0x52,0x80,0x7b]
          vminmaxpd  $123, -1024(%edx){1to4}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxpd  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vminmaxpd  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0xe5,0x0f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vminmaxpd  $123, (%eax){1to2}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x18,0x52,0x10,0x7b]
          vminmaxpd  $123, (%eax){1to2}, %xmm3, %xmm2

// CHECK: vminmaxpd  $123, -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x52,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxpd  $123, -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vminmaxpd  $123, 2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0x8f,0x52,0x51,0x7f,0x7b]
          vminmaxpd  $123, 2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxpd  $123, -1024(%edx){1to2}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0x9f,0x52,0x52,0x80,0x7b]
          vminmaxpd  $123, -1024(%edx){1to2}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxpd  $123, 268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x48,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxpd  $123, 268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vminmaxpd  $123, 291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0xe5,0x4f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxpd  $123, 291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vminmaxpd  $123, (%eax){1to8}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x58,0x52,0x10,0x7b]
          vminmaxpd  $123, (%eax){1to8}, %zmm3, %zmm2

// CHECK: vminmaxpd  $123, -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x48,0x52,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxpd  $123, -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vminmaxpd  $123, 8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0xcf,0x52,0x51,0x7f,0x7b]
          vminmaxpd  $123, 8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxpd  $123, -1024(%edx){1to8}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0xdf,0x52,0x52,0x80,0x7b]
          vminmaxpd  $123, -1024(%edx){1to8}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxph $123, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x52,0xd4,0x7b]
          vminmaxph $123, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxph $123, %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x64,0x0f,0x52,0xd4,0x7b]
          vminmaxph $123, %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vminmaxph $123, %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0x8f,0x52,0xd4,0x7b]
          vminmaxph $123, %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxph $123, %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x48,0x52,0xd4,0x7b]
          vminmaxph $123, %zmm4, %zmm3, %zmm2

// CHECK: vminmaxph $123, {sae}, %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x18,0x52,0xd4,0x7b]
          vminmaxph $123, {sae}, %zmm4, %zmm3, %zmm2

// CHECK: vminmaxph $123, %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x64,0x4f,0x52,0xd4,0x7b]
          vminmaxph $123, %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vminmaxph $123, {sae}, %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0x9f,0x52,0xd4,0x7b]
          vminmaxph $123, {sae}, %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxph $123, %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x64,0x28,0x52,0xd4,0x7b]
          vminmaxph $123, %ymm4, %ymm3, %ymm2

// CHECK: vminmaxph $123, {sae}, %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x60,0x18,0x52,0xd4,0x7b]
          vminmaxph $123, {sae}, %ymm4, %ymm3, %ymm2

// CHECK: vminmaxph $123, %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x64,0x2f,0x52,0xd4,0x7b]
          vminmaxph $123, %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vminmaxph $123, {sae}, %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x60,0x9f,0x52,0xd4,0x7b]
          vminmaxph $123, {sae}, %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxph  $123, 268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x64,0x28,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph  $123, 268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vminmaxph  $123, 291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x64,0x2f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph  $123, 291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vminmaxph  $123, (%eax){1to16}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x64,0x38,0x52,0x10,0x7b]
          vminmaxph  $123, (%eax){1to16}, %ymm3, %ymm2

// CHECK: vminmaxph  $123, -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x64,0x28,0x52,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxph  $123, -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vminmaxph  $123, 4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0xaf,0x52,0x51,0x7f,0x7b]
          vminmaxph  $123, 4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxph  $123, -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0xbf,0x52,0x52,0x80,0x7b]
          vminmaxph  $123, -256(%edx){1to16}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxph  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vminmaxph  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x64,0x0f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vminmaxph  $123, (%eax){1to8}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x18,0x52,0x10,0x7b]
          vminmaxph  $123, (%eax){1to8}, %xmm3, %xmm2

// CHECK: vminmaxph  $123, -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x52,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxph  $123, -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vminmaxph  $123, 2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0x8f,0x52,0x51,0x7f,0x7b]
          vminmaxph  $123, 2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxph  $123, -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0x9f,0x52,0x52,0x80,0x7b]
          vminmaxph  $123, -256(%edx){1to8}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxph  $123, 268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x48,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxph  $123, 268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vminmaxph  $123, 291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x64,0x4f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxph  $123, 291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vminmaxph  $123, (%eax){1to32}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x58,0x52,0x10,0x7b]
          vminmaxph  $123, (%eax){1to32}, %zmm3, %zmm2

// CHECK: vminmaxph  $123, -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x48,0x52,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxph  $123, -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vminmaxph  $123, 8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0xcf,0x52,0x51,0x7f,0x7b]
          vminmaxph  $123, 8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxph  $123, -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0xdf,0x52,0x52,0x80,0x7b]
          vminmaxph  $123, -256(%edx){1to32}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxps $123, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x52,0xd4,0x7b]
          vminmaxps $123, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxps $123, %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x65,0x0f,0x52,0xd4,0x7b]
          vminmaxps $123, %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vminmaxps $123, %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0x8f,0x52,0xd4,0x7b]
          vminmaxps $123, %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxps $123, %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x48,0x52,0xd4,0x7b]
          vminmaxps $123, %zmm4, %zmm3, %zmm2

// CHECK: vminmaxps $123, {sae}, %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x18,0x52,0xd4,0x7b]
          vminmaxps $123, {sae}, %zmm4, %zmm3, %zmm2

// CHECK: vminmaxps $123, %zmm4, %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x65,0x4f,0x52,0xd4,0x7b]
          vminmaxps $123, %zmm4, %zmm3, %zmm2 {%k7}

// CHECK: vminmaxps $123, {sae}, %zmm4, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0x9f,0x52,0xd4,0x7b]
          vminmaxps $123, {sae}, %zmm4, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxps $123, %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x65,0x28,0x52,0xd4,0x7b]
          vminmaxps $123, %ymm4, %ymm3, %ymm2

// CHECK: vminmaxps $123, {sae}, %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x61,0x18,0x52,0xd4,0x7b]
          vminmaxps $123, {sae}, %ymm4, %ymm3, %ymm2

// CHECK: vminmaxps $123, %ymm4, %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x65,0x2f,0x52,0xd4,0x7b]
          vminmaxps $123, %ymm4, %ymm3, %ymm2 {%k7}

// CHECK: vminmaxps $123, {sae}, %ymm4, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x61,0x9f,0x52,0xd4,0x7b]
          vminmaxps $123, {sae}, %ymm4, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxps  $123, 268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x65,0x28,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps  $123, 268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK: vminmaxps  $123, 291(%edi,%eax,4), %ymm3, %ymm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x65,0x2f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps  $123, 291(%edi,%eax,4), %ymm3, %ymm2 {%k7}

// CHECK: vminmaxps  $123, (%eax){1to8}, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x65,0x38,0x52,0x10,0x7b]
          vminmaxps  $123, (%eax){1to8}, %ymm3, %ymm2

// CHECK: vminmaxps  $123, -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf3,0x65,0x28,0x52,0x14,0x6d,0x00,0xfc,0xff,0xff,0x7b]
          vminmaxps  $123, -1024(,%ebp,2), %ymm3, %ymm2

// CHECK: vminmaxps  $123, 4064(%ecx), %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0xaf,0x52,0x51,0x7f,0x7b]
          vminmaxps  $123, 4064(%ecx), %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxps  $123, -512(%edx){1to8}, %ymm3, %ymm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0xbf,0x52,0x52,0x80,0x7b]
          vminmaxps  $123, -512(%edx){1to8}, %ymm3, %ymm2 {%k7} {z}

// CHECK: vminmaxps  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vminmaxps  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x65,0x0f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vminmaxps  $123, (%eax){1to4}, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x18,0x52,0x10,0x7b]
          vminmaxps  $123, (%eax){1to4}, %xmm3, %xmm2

// CHECK: vminmaxps  $123, -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x52,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
          vminmaxps  $123, -512(,%ebp,2), %xmm3, %xmm2

// CHECK: vminmaxps  $123, 2032(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0x8f,0x52,0x51,0x7f,0x7b]
          vminmaxps  $123, 2032(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxps  $123, -512(%edx){1to4}, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0x9f,0x52,0x52,0x80,0x7b]
          vminmaxps  $123, -512(%edx){1to4}, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxps  $123, 268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x48,0x52,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxps  $123, 268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK: vminmaxps  $123, 291(%edi,%eax,4), %zmm3, %zmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x65,0x4f,0x52,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxps  $123, 291(%edi,%eax,4), %zmm3, %zmm2 {%k7}

// CHECK: vminmaxps  $123, (%eax){1to16}, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x58,0x52,0x10,0x7b]
          vminmaxps  $123, (%eax){1to16}, %zmm3, %zmm2

// CHECK: vminmaxps  $123, -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x48,0x52,0x14,0x6d,0x00,0xf8,0xff,0xff,0x7b]
          vminmaxps  $123, -2048(,%ebp,2), %zmm3, %zmm2

// CHECK: vminmaxps  $123, 8128(%ecx), %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0xcf,0x52,0x51,0x7f,0x7b]
          vminmaxps  $123, 8128(%ecx), %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxps  $123, -512(%edx){1to16}, %zmm3, %zmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0xdf,0x52,0x52,0x80,0x7b]
          vminmaxps  $123, -512(%edx){1to16}, %zmm3, %zmm2 {%k7} {z}

// CHECK: vminmaxsd $123, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x53,0xd4,0x7b]
          vminmaxsd $123, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxsd $123, {sae}, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x18,0x53,0xd4,0x7b]
          vminmaxsd $123, {sae}, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxsd $123, %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0xe5,0x0f,0x53,0xd4,0x7b]
          vminmaxsd $123, %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vminmaxsd $123, {sae}, %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0x9f,0x53,0xd4,0x7b]
          vminmaxsd $123, {sae}, %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxsd  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x53,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxsd  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vminmaxsd  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0xe5,0x0f,0x53,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxsd  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vminmaxsd  $123, (%eax), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x53,0x10,0x7b]
          vminmaxsd  $123, (%eax), %xmm3, %xmm2

// CHECK: vminmaxsd  $123, -256(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0xe5,0x08,0x53,0x14,0x6d,0x00,0xff,0xff,0xff,0x7b]
          vminmaxsd  $123, -256(,%ebp,2), %xmm3, %xmm2

// CHECK: vminmaxsd  $123, 1016(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0x8f,0x53,0x51,0x7f,0x7b]
          vminmaxsd  $123, 1016(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxsd  $123, -1024(%edx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0xe5,0x8f,0x53,0x52,0x80,0x7b]
          vminmaxsd  $123, -1024(%edx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxsh $123, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x53,0xd4,0x7b]
          vminmaxsh $123, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxsh $123, {sae}, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x18,0x53,0xd4,0x7b]
          vminmaxsh $123, {sae}, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxsh $123, %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x64,0x0f,0x53,0xd4,0x7b]
          vminmaxsh $123, %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vminmaxsh $123, {sae}, %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0x9f,0x53,0xd4,0x7b]
          vminmaxsh $123, {sae}, %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxsh  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x53,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxsh  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vminmaxsh  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x64,0x0f,0x53,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxsh  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vminmaxsh  $123, (%eax), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x53,0x10,0x7b]
          vminmaxsh  $123, (%eax), %xmm3, %xmm2

// CHECK: vminmaxsh  $123, -64(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x64,0x08,0x53,0x14,0x6d,0xc0,0xff,0xff,0xff,0x7b]
          vminmaxsh  $123, -64(,%ebp,2), %xmm3, %xmm2

// CHECK: vminmaxsh  $123, 254(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0x8f,0x53,0x51,0x7f,0x7b]
          vminmaxsh  $123, 254(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxsh  $123, -256(%edx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x64,0x8f,0x53,0x52,0x80,0x7b]
          vminmaxsh  $123, -256(%edx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxss $123, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x53,0xd4,0x7b]
          vminmaxss $123, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxss $123, {sae}, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x18,0x53,0xd4,0x7b]
          vminmaxss $123, {sae}, %xmm4, %xmm3, %xmm2

// CHECK: vminmaxss $123, %xmm4, %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x65,0x0f,0x53,0xd4,0x7b]
          vminmaxss $123, %xmm4, %xmm3, %xmm2 {%k7}

// CHECK: vminmaxss $123, {sae}, %xmm4, %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0x9f,0x53,0xd4,0x7b]
          vminmaxss $123, {sae}, %xmm4, %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxss  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x53,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
          vminmaxss  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK: vminmaxss  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}
// CHECK: encoding: [0x62,0xf3,0x65,0x0f,0x53,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
          vminmaxss  $123, 291(%edi,%eax,4), %xmm3, %xmm2 {%k7}

// CHECK: vminmaxss  $123, (%eax), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x53,0x10,0x7b]
          vminmaxss  $123, (%eax), %xmm3, %xmm2

// CHECK: vminmaxss  $123, -128(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf3,0x65,0x08,0x53,0x14,0x6d,0x80,0xff,0xff,0xff,0x7b]
          vminmaxss  $123, -128(,%ebp,2), %xmm3, %xmm2

// CHECK: vminmaxss  $123, 508(%ecx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0x8f,0x53,0x51,0x7f,0x7b]
          vminmaxss  $123, 508(%ecx), %xmm3, %xmm2 {%k7} {z}

// CHECK: vminmaxss  $123, -512(%edx), %xmm3, %xmm2 {%k7} {z}
// CHECK: encoding: [0x62,0xf3,0x65,0x8f,0x53,0x52,0x80,0x7b]
          vminmaxss  $123, -512(%edx), %xmm3, %xmm2 {%k7} {z}

