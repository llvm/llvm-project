// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: tcvtrowd2ps zmm22, tmm5, ecx
// CHECK: encoding: [0x62,0xe2,0x76,0x48,0x4a,0xf5]
          tcvtrowd2ps zmm22, tmm5, ecx

// CHECK: tcvtrowd2ps zmm22, tmm2, ecx
// CHECK: encoding: [0x62,0xe2,0x76,0x48,0x4a,0xf2]
          tcvtrowd2ps zmm22, tmm2, ecx

// CHECK: tcvtrowd2ps zmm22, tmm5, 123
// CHECK: encoding: [0x62,0xe3,0x7e,0x48,0x07,0xf5,0x7b]
          tcvtrowd2ps zmm22, tmm5, 123

// CHECK: tcvtrowd2ps zmm22, tmm2, 123
// CHECK: encoding: [0x62,0xe3,0x7e,0x48,0x07,0xf2,0x7b]
          tcvtrowd2ps zmm22, tmm2, 123

// CHECK: tcvtrowps2bf16h zmm22, tmm5, ecx
// CHECK: encoding: [0x62,0xe2,0x77,0x48,0x6d,0xf5]
          tcvtrowps2bf16h zmm22, tmm5, ecx

// CHECK: tcvtrowps2bf16h zmm22, tmm2, ecx
// CHECK: encoding: [0x62,0xe2,0x77,0x48,0x6d,0xf2]
          tcvtrowps2bf16h zmm22, tmm2, ecx

// CHECK: tcvtrowps2bf16h zmm22, tmm5, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x07,0xf5,0x7b]
          tcvtrowps2bf16h zmm22, tmm5, 123

// CHECK: tcvtrowps2bf16h zmm22, tmm2, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x07,0xf2,0x7b]
          tcvtrowps2bf16h zmm22, tmm2, 123

// CHECK: tcvtrowps2bf16l zmm22, tmm5, ecx
// CHECK: encoding: [0x62,0xe2,0x76,0x48,0x6d,0xf5]
          tcvtrowps2bf16l zmm22, tmm5, ecx

// CHECK: tcvtrowps2bf16l zmm22, tmm2, ecx
// CHECK: encoding: [0x62,0xe2,0x76,0x48,0x6d,0xf2]
          tcvtrowps2bf16l zmm22, tmm2, ecx

// CHECK: tcvtrowps2bf16l zmm22, tmm5, 123
// CHECK: encoding: [0x62,0xe3,0x7e,0x48,0x77,0xf5,0x7b]
          tcvtrowps2bf16l zmm22, tmm5, 123

// CHECK: tcvtrowps2bf16l zmm22, tmm2, 123
// CHECK: encoding: [0x62,0xe3,0x7e,0x48,0x77,0xf2,0x7b]
          tcvtrowps2bf16l zmm22, tmm2, 123

// CHECK: tcvtrowps2phh zmm22, tmm5, ecx
// CHECK: encoding: [0x62,0xe2,0x74,0x48,0x6d,0xf5]
          tcvtrowps2phh zmm22, tmm5, ecx

// CHECK: tcvtrowps2phh zmm22, tmm2, ecx
// CHECK: encoding: [0x62,0xe2,0x74,0x48,0x6d,0xf2]
          tcvtrowps2phh zmm22, tmm2, ecx

// CHECK: tcvtrowps2phh zmm22, tmm5, 123
// CHECK: encoding: [0x62,0xe3,0x7c,0x48,0x07,0xf5,0x7b]
          tcvtrowps2phh zmm22, tmm5, 123

// CHECK: tcvtrowps2phh zmm22, tmm2, 123
// CHECK: encoding: [0x62,0xe3,0x7c,0x48,0x07,0xf2,0x7b]
          tcvtrowps2phh zmm22, tmm2, 123

// CHECK: tcvtrowps2phl zmm22, tmm5, ecx
// CHECK: encoding: [0x62,0xe2,0x75,0x48,0x6d,0xf5]
          tcvtrowps2phl zmm22, tmm5, ecx

// CHECK: tcvtrowps2phl zmm22, tmm2, ecx
// CHECK: encoding: [0x62,0xe2,0x75,0x48,0x6d,0xf2]
          tcvtrowps2phl zmm22, tmm2, ecx

// CHECK: tcvtrowps2phl zmm22, tmm5, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x77,0xf5,0x7b]
          tcvtrowps2phl zmm22, tmm5, 123

// CHECK: tcvtrowps2phl zmm22, tmm2, 123
// CHECK: encoding: [0x62,0xe3,0x7f,0x48,0x77,0xf2,0x7b]
          tcvtrowps2phl zmm22, tmm2, 123

// CHECK: tilemovrow zmm22, tmm3, ecx
// CHECK: encoding: [0x62,0xe2,0x75,0x48,0x4a,0xf3]
          tilemovrow zmm22, tmm3, ecx

// CHECK: tilemovrow zmm22, tmm2, ecx
// CHECK: encoding: [0x62,0xe2,0x75,0x48,0x4a,0xf2]
          tilemovrow zmm22, tmm2, ecx

// CHECK: tilemovrow zmm22, tmm3, 123
// CHECK: encoding: [0x62,0xe3,0x7d,0x48,0x07,0xf3,0x7b]
          tilemovrow zmm22, tmm3, 123

// CHECK: tilemovrow zmm22, tmm2, 123
// CHECK: encoding: [0x62,0xe3,0x7d,0x48,0x07,0xf2,0x7b]
          tilemovrow zmm22, tmm2, 123

// CHECK: tilemovrow zmm22, tmm0, edx
// CHECK: encoding: [0x62,0xe2,0x6d,0x48,0x4a,0xf0]
          tilemovrow zmm22, tmm0, edx

// CHECK: tilemovrow zmm22, tmm0, 123
// CHECK: encoding: [0x62,0xe3,0x7d,0x48,0x07,0xf0,0x7b]
          tilemovrow zmm22, tmm0, 123
