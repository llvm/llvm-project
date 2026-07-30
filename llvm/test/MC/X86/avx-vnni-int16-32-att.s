// RUN: llvm-mc -triple i686-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK:      vpdpwsud %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0xd4]
               vpdpwsud %ymm4, %ymm3, %ymm2

// CHECK:      vpdpwsud %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0xd4]
               vpdpwsud %xmm4, %xmm3, %xmm2

// CHECK:      vpdpwsud  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwsud  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK:      vpdpwsud  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwsud  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK:      vpdpwsud  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0x10]
               vpdpwsud  (%eax), %ymm3, %ymm2

// CHECK:      vpdpwsud  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vpdpwsud  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK:      vpdpwsud  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0x91,0xe0,0x0f,0x00,0x00]
               vpdpwsud  4064(%ecx), %ymm3, %ymm2

// CHECK:      vpdpwsud  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd2,0x92,0x00,0xf0,0xff,0xff]
               vpdpwsud  -4096(%edx), %ymm3, %ymm2

// CHECK:      vpdpwsud  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwsud  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      vpdpwsud  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwsud  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      vpdpwsud  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0x10]
               vpdpwsud  (%eax), %xmm3, %xmm2

// CHECK:      vpdpwsud  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vpdpwsud  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      vpdpwsud  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0x91,0xf0,0x07,0x00,0x00]
               vpdpwsud  2032(%ecx), %xmm3, %xmm2

// CHECK:      vpdpwsud  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd2,0x92,0x00,0xf8,0xff,0xff]
               vpdpwsud  -2048(%edx), %xmm3, %xmm2

// CHECK:      vpdpwsuds %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0xd4]
               vpdpwsuds %ymm4, %ymm3, %ymm2

// CHECK:      vpdpwsuds %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0xd4]
               vpdpwsuds %xmm4, %xmm3, %xmm2

// CHECK:      vpdpwsuds  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwsuds  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK:      vpdpwsuds  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwsuds  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK:      vpdpwsuds  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0x10]
               vpdpwsuds  (%eax), %ymm3, %ymm2

// CHECK:      vpdpwsuds  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vpdpwsuds  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK:      vpdpwsuds  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0x91,0xe0,0x0f,0x00,0x00]
               vpdpwsuds  4064(%ecx), %ymm3, %ymm2

// CHECK:      vpdpwsuds  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x66,0xd3,0x92,0x00,0xf0,0xff,0xff]
               vpdpwsuds  -4096(%edx), %ymm3, %ymm2

// CHECK:      vpdpwsuds  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwsuds  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      vpdpwsuds  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwsuds  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      vpdpwsuds  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0x10]
               vpdpwsuds  (%eax), %xmm3, %xmm2

// CHECK:      vpdpwsuds  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vpdpwsuds  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      vpdpwsuds  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0x91,0xf0,0x07,0x00,0x00]
               vpdpwsuds  2032(%ecx), %xmm3, %xmm2

// CHECK:      vpdpwsuds  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x62,0xd3,0x92,0x00,0xf8,0xff,0xff]
               vpdpwsuds  -2048(%edx), %xmm3, %xmm2

// CHECK:      vpdpwusd %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0xd4]
               vpdpwusd %ymm4, %ymm3, %ymm2

// CHECK:      vpdpwusd %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0xd4]
               vpdpwusd %xmm4, %xmm3, %xmm2

// CHECK:      vpdpwusd  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwusd  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK:      vpdpwusd  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwusd  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK:      vpdpwusd  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0x10]
               vpdpwusd  (%eax), %ymm3, %ymm2

// CHECK:      vpdpwusd  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vpdpwusd  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK:      vpdpwusd  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0x91,0xe0,0x0f,0x00,0x00]
               vpdpwusd  4064(%ecx), %ymm3, %ymm2

// CHECK:      vpdpwusd  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd2,0x92,0x00,0xf0,0xff,0xff]
               vpdpwusd  -4096(%edx), %ymm3, %ymm2

// CHECK:      vpdpwusd  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwusd  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      vpdpwusd  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwusd  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      vpdpwusd  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0x10]
               vpdpwusd  (%eax), %xmm3, %xmm2

// CHECK:      vpdpwusd  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vpdpwusd  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      vpdpwusd  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0x91,0xf0,0x07,0x00,0x00]
               vpdpwusd  2032(%ecx), %xmm3, %xmm2

// CHECK:      vpdpwusd  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd2,0x92,0x00,0xf8,0xff,0xff]
               vpdpwusd  -2048(%edx), %xmm3, %xmm2

// CHECK:      vpdpwusds %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0xd4]
               vpdpwusds %ymm4, %ymm3, %ymm2

// CHECK:      vpdpwusds %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0xd4]
               vpdpwusds %xmm4, %xmm3, %xmm2

// CHECK:      vpdpwusds  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwusds  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK:      vpdpwusds  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwusds  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK:      vpdpwusds  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0x10]
               vpdpwusds  (%eax), %ymm3, %ymm2

// CHECK:      vpdpwusds  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vpdpwusds  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK:      vpdpwusds  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0x91,0xe0,0x0f,0x00,0x00]
               vpdpwusds  4064(%ecx), %ymm3, %ymm2

// CHECK:      vpdpwusds  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x65,0xd3,0x92,0x00,0xf0,0xff,0xff]
               vpdpwusds  -4096(%edx), %ymm3, %ymm2

// CHECK:      vpdpwusds  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwusds  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      vpdpwusds  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwusds  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      vpdpwusds  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0x10]
               vpdpwusds  (%eax), %xmm3, %xmm2

// CHECK:      vpdpwusds  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vpdpwusds  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      vpdpwusds  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0x91,0xf0,0x07,0x00,0x00]
               vpdpwusds  2032(%ecx), %xmm3, %xmm2

// CHECK:      vpdpwusds  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xd3,0x92,0x00,0xf8,0xff,0xff]
               vpdpwusds  -2048(%edx), %xmm3, %xmm2

// CHECK:      vpdpwuud %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0xd4]
               vpdpwuud %ymm4, %ymm3, %ymm2

// CHECK:      vpdpwuud %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0xd4]
               vpdpwuud %xmm4, %xmm3, %xmm2

// CHECK:      vpdpwuud  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwuud  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK:      vpdpwuud  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwuud  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK:      vpdpwuud  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0x10]
               vpdpwuud  (%eax), %ymm3, %ymm2

// CHECK:      vpdpwuud  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vpdpwuud  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK:      vpdpwuud  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0x91,0xe0,0x0f,0x00,0x00]
               vpdpwuud  4064(%ecx), %ymm3, %ymm2

// CHECK:      vpdpwuud  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd2,0x92,0x00,0xf0,0xff,0xff]
               vpdpwuud  -4096(%edx), %ymm3, %ymm2

// CHECK:      vpdpwuud  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwuud  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      vpdpwuud  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwuud  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      vpdpwuud  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0x10]
               vpdpwuud  (%eax), %xmm3, %xmm2

// CHECK:      vpdpwuud  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vpdpwuud  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      vpdpwuud  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0x91,0xf0,0x07,0x00,0x00]
               vpdpwuud  2032(%ecx), %xmm3, %xmm2

// CHECK:      vpdpwuud  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd2,0x92,0x00,0xf8,0xff,0xff]
               vpdpwuud  -2048(%edx), %xmm3, %xmm2

// CHECK:      vpdpwuuds %ymm4, %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0xd4]
               vpdpwuuds %ymm4, %ymm3, %ymm2

// CHECK:      vpdpwuuds %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0xd4]
               vpdpwuuds %xmm4, %xmm3, %xmm2

// CHECK:      vpdpwuuds  268435456(%esp,%esi,8), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwuuds  268435456(%esp,%esi,8), %ymm3, %ymm2

// CHECK:      vpdpwuuds  291(%edi,%eax,4), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwuuds  291(%edi,%eax,4), %ymm3, %ymm2

// CHECK:      vpdpwuuds  (%eax), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0x10]
               vpdpwuuds  (%eax), %ymm3, %ymm2

// CHECK:      vpdpwuuds  -1024(,%ebp,2), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0x14,0x6d,0x00,0xfc,0xff,0xff]
               vpdpwuuds  -1024(,%ebp,2), %ymm3, %ymm2

// CHECK:      vpdpwuuds  4064(%ecx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0x91,0xe0,0x0f,0x00,0x00]
               vpdpwuuds  4064(%ecx), %ymm3, %ymm2

// CHECK:      vpdpwuuds  -4096(%edx), %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x64,0xd3,0x92,0x00,0xf0,0xff,0xff]
               vpdpwuuds  -4096(%edx), %ymm3, %ymm2

// CHECK:      vpdpwuuds  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0x94,0xf4,0x00,0x00,0x00,0x10]
               vpdpwuuds  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      vpdpwuuds  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0x94,0x87,0x23,0x01,0x00,0x00]
               vpdpwuuds  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      vpdpwuuds  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0x10]
               vpdpwuuds  (%eax), %xmm3, %xmm2

// CHECK:      vpdpwuuds  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vpdpwuuds  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      vpdpwuuds  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0x91,0xf0,0x07,0x00,0x00]
               vpdpwuuds  2032(%ecx), %xmm3, %xmm2

// CHECK:      vpdpwuuds  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xd3,0x92,0x00,0xf8,0xff,0xff]
               vpdpwuuds  -2048(%edx), %xmm3, %xmm2

