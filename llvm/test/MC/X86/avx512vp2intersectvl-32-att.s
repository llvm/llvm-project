// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: vp2intersectd %ymm4, %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0xf4]
          vp2intersectd %ymm4, %ymm3, %k6

// CHECK: vp2intersectd %xmm4, %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0xf4]
          vp2intersectd %xmm4, %xmm3, %k6

// CHECK: vp2intersectd  268435456(%esp,%esi,8), %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectd  268435456(%esp,%esi,8), %ymm3, %k6

// CHECK: vp2intersectd  291(%edi,%eax,4), %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0xb4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectd  291(%edi,%eax,4), %ymm3, %k6

// CHECK: vp2intersectd  (%eax){1to8}, %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x38,0x68,0x30]
          vp2intersectd  (%eax){1to8}, %ymm3, %k6

// CHECK: vp2intersectd  -1024(,%ebp,2), %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vp2intersectd  -1024(,%ebp,2), %ymm3, %k6

// CHECK: vp2intersectd  4064(%ecx), %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x28,0x68,0x71,0x7f]
          vp2intersectd  4064(%ecx), %ymm3, %k6

// CHECK: vp2intersectd  -512(%edx){1to8}, %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x38,0x68,0x72,0x80]
          vp2intersectd  -512(%edx){1to8}, %ymm3, %k6

// CHECK: vp2intersectd  268435456(%esp,%esi,8), %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectd  268435456(%esp,%esi,8), %xmm3, %k6

// CHECK: vp2intersectd  291(%edi,%eax,4), %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0xb4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectd  291(%edi,%eax,4), %xmm3, %k6

// CHECK: vp2intersectd  (%eax){1to4}, %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x18,0x68,0x30]
          vp2intersectd  (%eax){1to4}, %xmm3, %k6

// CHECK: vp2intersectd  -512(,%ebp,2), %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vp2intersectd  -512(,%ebp,2), %xmm3, %k6

// CHECK: vp2intersectd  2032(%ecx), %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x08,0x68,0x71,0x7f]
          vp2intersectd  2032(%ecx), %xmm3, %k6

// CHECK: vp2intersectd  -512(%edx){1to4}, %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0x67,0x18,0x68,0x72,0x80]
          vp2intersectd  -512(%edx){1to4}, %xmm3, %k6

// CHECK: vp2intersectq %ymm4, %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0xf4]
          vp2intersectq %ymm4, %ymm3, %k6

// CHECK: vp2intersectq %xmm4, %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0xf4]
          vp2intersectq %xmm4, %xmm3, %k6

// CHECK: vp2intersectq  268435456(%esp,%esi,8), %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectq  268435456(%esp,%esi,8), %ymm3, %k6

// CHECK: vp2intersectq  291(%edi,%eax,4), %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0xb4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectq  291(%edi,%eax,4), %ymm3, %k6

// CHECK: vp2intersectq  (%eax){1to4}, %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x38,0x68,0x30]
          vp2intersectq  (%eax){1to4}, %ymm3, %k6

// CHECK: vp2intersectq  -1024(,%ebp,2), %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0x34,0x6d,0x00,0xfc,0xff,0xff]
          vp2intersectq  -1024(,%ebp,2), %ymm3, %k6

// CHECK: vp2intersectq  4064(%ecx), %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x28,0x68,0x71,0x7f]
          vp2intersectq  4064(%ecx), %ymm3, %k6

// CHECK: vp2intersectq  -1024(%edx){1to4}, %ymm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x38,0x68,0x72,0x80]
          vp2intersectq  -1024(%edx){1to4}, %ymm3, %k6

// CHECK: vp2intersectq  268435456(%esp,%esi,8), %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vp2intersectq  268435456(%esp,%esi,8), %xmm3, %k6

// CHECK: vp2intersectq  291(%edi,%eax,4), %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0xb4,0x87,0x23,0x01,0x00,0x00]
          vp2intersectq  291(%edi,%eax,4), %xmm3, %k6

// CHECK: vp2intersectq  (%eax){1to2}, %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x18,0x68,0x30]
          vp2intersectq  (%eax){1to2}, %xmm3, %k6

// CHECK: vp2intersectq  -512(,%ebp,2), %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0x34,0x6d,0x00,0xfe,0xff,0xff]
          vp2intersectq  -512(,%ebp,2), %xmm3, %k6

// CHECK: vp2intersectq  2032(%ecx), %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x08,0x68,0x71,0x7f]
          vp2intersectq  2032(%ecx), %xmm3, %k6

// CHECK: vp2intersectq  -1024(%edx){1to2}, %xmm3, %k6
// CHECK: encoding: [0x62,0xf2,0xe7,0x18,0x68,0x72,0x80]
          vp2intersectq  -1024(%edx){1to2}, %xmm3, %k6
