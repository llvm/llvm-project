// RUN: llvm-mc -triple i686-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK:      vsm3msg1 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xda,0xd4]
               vsm3msg1 %xmm4, %xmm3, %xmm2

// CHECK:      vsm3msg1  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               vsm3msg1  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      vsm3msg1  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               vsm3msg1  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      vsm3msg1  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xda,0x10]
               vsm3msg1  (%eax), %xmm3, %xmm2

// CHECK:      vsm3msg1  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xda,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vsm3msg1  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      vsm3msg1  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xda,0x91,0xf0,0x07,0x00,0x00]
               vsm3msg1  2032(%ecx), %xmm3, %xmm2

// CHECK:      vsm3msg1  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x60,0xda,0x92,0x00,0xf8,0xff,0xff]
               vsm3msg1  -2048(%edx), %xmm3, %xmm2

// CHECK:      vsm3msg2 %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xda,0xd4]
               vsm3msg2 %xmm4, %xmm3, %xmm2

// CHECK:      vsm3msg2  268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xda,0x94,0xf4,0x00,0x00,0x00,0x10]
               vsm3msg2  268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      vsm3msg2  291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xda,0x94,0x87,0x23,0x01,0x00,0x00]
               vsm3msg2  291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      vsm3msg2  (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xda,0x10]
               vsm3msg2  (%eax), %xmm3, %xmm2

// CHECK:      vsm3msg2  -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xda,0x14,0x6d,0x00,0xfe,0xff,0xff]
               vsm3msg2  -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      vsm3msg2  2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xda,0x91,0xf0,0x07,0x00,0x00]
               vsm3msg2  2032(%ecx), %xmm3, %xmm2

// CHECK:      vsm3msg2  -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe2,0x61,0xda,0x92,0x00,0xf8,0xff,0xff]
               vsm3msg2  -2048(%edx), %xmm3, %xmm2

// CHECK:      vsm3rnds2 $123, %xmm4, %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe3,0x61,0xde,0xd4,0x7b]
               vsm3rnds2 $123, %xmm4, %xmm3, %xmm2

// CHECK:      vsm3rnds2  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe3,0x61,0xde,0x94,0xf4,0x00,0x00,0x00,0x10,0x7b]
               vsm3rnds2  $123, 268435456(%esp,%esi,8), %xmm3, %xmm2

// CHECK:      vsm3rnds2  $123, 291(%edi,%eax,4), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe3,0x61,0xde,0x94,0x87,0x23,0x01,0x00,0x00,0x7b]
               vsm3rnds2  $123, 291(%edi,%eax,4), %xmm3, %xmm2

// CHECK:      vsm3rnds2  $123, (%eax), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe3,0x61,0xde,0x10,0x7b]
               vsm3rnds2  $123, (%eax), %xmm3, %xmm2

// CHECK:      vsm3rnds2  $123, -512(,%ebp,2), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe3,0x61,0xde,0x14,0x6d,0x00,0xfe,0xff,0xff,0x7b]
               vsm3rnds2  $123, -512(,%ebp,2), %xmm3, %xmm2

// CHECK:      vsm3rnds2  $123, 2032(%ecx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe3,0x61,0xde,0x91,0xf0,0x07,0x00,0x00,0x7b]
               vsm3rnds2  $123, 2032(%ecx), %xmm3, %xmm2

// CHECK:      vsm3rnds2  $123, -2048(%edx), %xmm3, %xmm2
// CHECK: encoding: [0xc4,0xe3,0x61,0xde,0x92,0x00,0xf8,0xff,0xff,0x7b]
               vsm3rnds2  $123, -2048(%edx), %xmm3, %xmm2

