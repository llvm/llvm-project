// RUN: llvm-mc -triple i686-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK:      vsm4key4 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0xd4]
               vsm4key4 %zmm4, %zmm3, %zmm2

// CHECK:      vsm4key4  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               vsm4key4  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK:      vsm4key4  291(%edi,%eax,4), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               vsm4key4  291(%edi,%eax,4), %zmm3, %zmm2

// CHECK:      vsm4key4  (%eax), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x10]
               vsm4key4  (%eax), %zmm3, %zmm2

// CHECK:      vsm4key4  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x14,0x6d,0x00,0xf8,0xff,0xff]
               vsm4key4  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK:      vsm4key4  8128(%ecx), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x51,0x7f]
               vsm4key4  8128(%ecx), %zmm3, %zmm2

// CHECK:      vsm4key4  -8192(%edx), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x48,0xda,0x52,0x80]
               vsm4key4  -8192(%edx), %zmm3, %zmm2

// CHECK:      vsm4rnds4 %zmm4, %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0xd4]
               vsm4rnds4 %zmm4, %zmm3, %zmm2

// CHECK:      vsm4rnds4  268435456(%esp,%esi,8), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               vsm4rnds4  268435456(%esp,%esi,8), %zmm3, %zmm2

// CHECK:      vsm4rnds4  291(%edi,%eax,4), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               vsm4rnds4  291(%edi,%eax,4), %zmm3, %zmm2

// CHECK:      vsm4rnds4  (%eax), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x10]
               vsm4rnds4  (%eax), %zmm3, %zmm2

// CHECK:      vsm4rnds4  -2048(,%ebp,2), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x14,0x6d,0x00,0xf8,0xff,0xff]
               vsm4rnds4  -2048(,%ebp,2), %zmm3, %zmm2

// CHECK:      vsm4rnds4  8128(%ecx), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x51,0x7f]
               vsm4rnds4  8128(%ecx), %zmm3, %zmm2

// CHECK:      vsm4rnds4  -8192(%edx), %zmm3, %zmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x48,0xda,0x52,0x80]
               vsm4rnds4  -8192(%edx), %zmm3, %zmm2

// CHECK:      {evex} vsm4key4 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0xd4]
               {evex} vsm4key4 %ymm4, %ymm3, %ymm2

// CHECK:      {evex} vsm4key4 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0xd4]
               {evex} vsm4key4 %xmm4, %xmm3, %xmm2

// CHECK:      {evex} vsm4key4  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               {evex} vsm4key4  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK:      {evex} vsm4key4  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               {evex} vsm4key4  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK:      {evex} vsm4key4  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x10]
               {evex} vsm4key4  (%eax), %ymm3, %ymm2

// CHECK:      {evex} vsm4key4  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x14,0x6d,0x00,0xfc,0xff,0xff]
               {evex} vsm4key4  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK:      {evex} vsm4key4  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x51,0x7f]
               {evex} vsm4key4  4064(%ecx), %ymm3, %ymm2

// CHECK:      {evex} vsm4key4  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x66,0x28,0xda,0x52,0x80]
               {evex} vsm4key4  -4096(%edx), %ymm3, %ymm2

// CHECK:      {evex} vsm4key4  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               {evex} vsm4key4  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      {evex} vsm4key4  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               {evex} vsm4key4  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      {evex} vsm4key4  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x10]
               {evex} vsm4key4  (%eax), %xmm3, %xmm2

// CHECK:      {evex} vsm4key4  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x14,0x6d,0x00,0xfe,0xff,0xff]
               {evex} vsm4key4  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      {evex} vsm4key4  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x51,0x7f]
               {evex} vsm4key4  2032(%ecx), %xmm3, %xmm2

// CHECK:      {evex} vsm4key4  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x66,0x08,0xda,0x52,0x80]
               {evex} vsm4key4  -2048(%edx), %xmm3, %xmm2

// CHECK:      {evex} vsm4rnds4 %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0xd4]
               {evex} vsm4rnds4 %ymm4, %ymm3, %ymm2

// CHECK:      {evex} vsm4rnds4 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0xd4]
               {evex} vsm4rnds4 %xmm4, %xmm3, %xmm2

// CHECK:      {evex} vsm4rnds4  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               {evex} vsm4rnds4  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK:      {evex} vsm4rnds4  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               {evex} vsm4rnds4  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK:      {evex} vsm4rnds4  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x10]
               {evex} vsm4rnds4  (%eax), %ymm3, %ymm2

// CHECK:      {evex} vsm4rnds4  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x14,0x6d,0x00,0xfc,0xff,0xff]
               {evex} vsm4rnds4  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK:      {evex} vsm4rnds4  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x51,0x7f]
               {evex} vsm4rnds4  4064(%ecx), %ymm3, %ymm2

// CHECK:      {evex} vsm4rnds4  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0xda,0x52,0x80]
               {evex} vsm4rnds4  -4096(%edx), %ymm3, %ymm2

// CHECK:      {evex} vsm4rnds4  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               {evex} vsm4rnds4  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      {evex} vsm4rnds4  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               {evex} vsm4rnds4  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      {evex} vsm4rnds4  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x10]
               {evex} vsm4rnds4  (%eax), %xmm3, %xmm2

// CHECK:      {evex} vsm4rnds4  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x14,0x6d,0x00,0xfe,0xff,0xff]
               {evex} vsm4rnds4  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      {evex} vsm4rnds4  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x51,0x7f]
               {evex} vsm4rnds4  2032(%ecx), %xmm3, %xmm2

// CHECK:      {evex} vsm4rnds4  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0xda,0x52,0x80]
               {evex} vsm4rnds4  -2048(%edx), %xmm3, %xmm2