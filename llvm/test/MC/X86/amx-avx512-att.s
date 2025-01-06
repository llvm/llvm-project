// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s  | FileCheck %s

// CHECK: tcvtrowd2ps %ecx, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe2,0x76,0x48,0x4a,0xf5]
          tcvtrowd2ps %ecx, %tmm5, %zmm22

// CHECK: tcvtrowd2ps %ecx, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe2,0x76,0x48,0x4a,0xf2]
          tcvtrowd2ps %ecx, %tmm2, %zmm22

// CHECK: tcvtrowd2ps $123, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7e,0x48,0x07,0xf5,0x7b]
          tcvtrowd2ps $123, %tmm5, %zmm22

// CHECK: tcvtrowd2ps $123, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7e,0x48,0x07,0xf2,0x7b]
          tcvtrowd2ps $123, %tmm2, %zmm22

// CHECK: tcvtrowps2pbf16h %ecx, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe2,0x77,0x48,0x6d,0xf5]
          tcvtrowps2pbf16h %ecx, %tmm5, %zmm22

// CHECK: tcvtrowps2pbf16h %ecx, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe2,0x77,0x48,0x6d,0xf2]
          tcvtrowps2pbf16h %ecx, %tmm2, %zmm22

// CHECK: tcvtrowps2pbf16h $123, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x07,0xf5,0x7b]
          tcvtrowps2pbf16h $123, %tmm5, %zmm22

// CHECK: tcvtrowps2pbf16h $123, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x07,0xf2,0x7b]
          tcvtrowps2pbf16h $123, %tmm2, %zmm22

// CHECK: tcvtrowps2pbf16l %ecx, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe2,0x76,0x48,0x6d,0xf5]
          tcvtrowps2pbf16l %ecx, %tmm5, %zmm22

// CHECK: tcvtrowps2pbf16l %ecx, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe2,0x76,0x48,0x6d,0xf2]
          tcvtrowps2pbf16l %ecx, %tmm2, %zmm22

// CHECK: tcvtrowps2pbf16l $123, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7e,0x48,0x77,0xf5,0x7b]
          tcvtrowps2pbf16l $123, %tmm5, %zmm22

// CHECK: tcvtrowps2pbf16l $123, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7e,0x48,0x77,0xf2,0x7b]
          tcvtrowps2pbf16l $123, %tmm2, %zmm22

// CHECK: tcvtrowps2phh %ecx, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe2,0x74,0x48,0x6d,0xf5]
          tcvtrowps2phh %ecx, %tmm5, %zmm22

// CHECK: tcvtrowps2phh %ecx, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe2,0x74,0x48,0x6d,0xf2]
          tcvtrowps2phh %ecx, %tmm2, %zmm22

// CHECK: tcvtrowps2phh $123, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7c,0x48,0x07,0xf5,0x7b]
          tcvtrowps2phh $123, %tmm5, %zmm22

// CHECK: tcvtrowps2phh $123, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7c,0x48,0x07,0xf2,0x7b]
          tcvtrowps2phh $123, %tmm2, %zmm22

// CHECK: tcvtrowps2phl %ecx, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe2,0x75,0x48,0x6d,0xf5]
          tcvtrowps2phl %ecx, %tmm5, %zmm22

// CHECK: tcvtrowps2phl %ecx, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe2,0x75,0x48,0x6d,0xf2]
          tcvtrowps2phl %ecx, %tmm2, %zmm22

// CHECK: tcvtrowps2phl $123, %tmm5, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x77,0xf5,0x7b]
          tcvtrowps2phl $123, %tmm5, %zmm22

// CHECK: tcvtrowps2phl $123, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x77,0xf2,0x7b]
          tcvtrowps2phl $123, %tmm2, %zmm22

// CHECK: tilemovrow %ecx, %tmm3, %zmm22
// CHECK: encoding: [0x62,0xe2,0x75,0x48,0x4a,0xf3]
          tilemovrow %ecx, %tmm3, %zmm22

// CHECK: tilemovrow %ecx, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe2,0x75,0x48,0x4a,0xf2]
          tilemovrow %ecx, %tmm2, %zmm22

// CHECK: tilemovrow $123, %tmm3, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7d,0x48,0x07,0xf3,0x7b]
          tilemovrow $123, %tmm3, %zmm22

// CHECK: tilemovrow $123, %tmm2, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7d,0x48,0x07,0xf2,0x7b]
          tilemovrow $123, %tmm2, %zmm22

// CHECK: tilemovrow %edx, %tmm0, %zmm22
// CHECK: encoding: [0x62,0xe2,0x6d,0x48,0x4a,0xf0]
          tilemovrow %edx, %tmm0, %zmm22

// CHECK: tilemovrow $123, %tmm0, %zmm22
// CHECK: encoding: [0x62,0xe3,0x7d,0x48,0x07,0xf0,0x7b]
          tilemovrow $123, %tmm0, %zmm22
