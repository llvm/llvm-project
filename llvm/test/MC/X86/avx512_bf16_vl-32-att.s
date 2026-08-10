// RUN: llvm-mc -triple i686-unknown-unknown --show-encoding < %s | FileCheck %s

// CHECK: vcvtne2ps2bf16 %xmm4, %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0xf4]
          vcvtne2ps2bf16 %xmm4, %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16 %xmm4, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x57,0x8f,0x72,0xf4]
          vcvtne2ps2bf16 %xmm4, %xmm5, %xmm6 {%k7} {z}

// CHECK: vcvtne2ps2bf16   (%ecx), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0x31]
          vcvtne2ps2bf16   (%ecx), %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16   291(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0xb4,0xf4,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16   291(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16   268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16   268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16   -16(%esp), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0x74,0x24,0xff]
          vcvtne2ps2bf16   -16(%esp), %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16   (%eax){1to4}, %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x1f,0x72,0x30]
          vcvtne2ps2bf16   (%eax){1to4}, %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16   2032(%edx), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0x72,0x7f]
          vcvtne2ps2bf16   2032(%edx), %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16   -2048(%edx), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x0f,0x72,0x72,0x80]
          vcvtne2ps2bf16   -2048(%edx), %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16   508(%edx){1to4}, %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x1f,0x72,0x72,0x7f]
          vcvtne2ps2bf16   508(%edx){1to4}, %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16   -512(%edx){1to4}, %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x1f,0x72,0x72,0x80]
          vcvtne2ps2bf16   -512(%edx){1to4}, %xmm5, %xmm6 {%k7}

// CHECK: vcvtne2ps2bf16 %ymm4, %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0xf4]
          vcvtne2ps2bf16 %ymm4, %ymm5, %ymm6 {%k7}

// CHECK: vcvtne2ps2bf16 %ymm4, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x57,0xaf,0x72,0xf4]
          vcvtne2ps2bf16 %ymm4, %ymm5, %ymm6 {%k7} {z}

// CHECK: vcvtne2ps2bf16   (%ecx), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0x31]
          vcvtne2ps2bf16   (%ecx), %ymm5, %ymm6 {%k7}

// CHECK: vcvtne2ps2bf16   291(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0xb4,0xf4,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16   291(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vcvtne2ps2bf16   268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16   268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vcvtne2ps2bf16   -32(%esp), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0x74,0x24,0xff]
          vcvtne2ps2bf16   -32(%esp), %ymm5, %ymm6 {%k7}

// CHECK: vcvtne2ps2bf16   (%eax){1to8}, %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x3f,0x72,0x30]
          vcvtne2ps2bf16   (%eax){1to8}, %ymm5, %ymm6 {%k7}

// CHECK: vcvtne2ps2bf16   4064(%edx), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0x72,0x7f]
          vcvtne2ps2bf16   4064(%edx), %ymm5, %ymm6 {%k7}

// CHECK: vcvtne2ps2bf16   -4096(%edx), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x2f,0x72,0x72,0x80]
          vcvtne2ps2bf16   -4096(%edx), %ymm5, %ymm6 {%k7}

// CHECK: vcvtne2ps2bf16   508(%edx){1to8}, %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x3f,0x72,0x72,0x7f]
          vcvtne2ps2bf16   508(%edx){1to8}, %ymm5, %ymm6 {%k7}

// CHECK: vcvtne2ps2bf16   -512(%edx){1to8}, %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x3f,0x72,0x72,0x80]
          vcvtne2ps2bf16   -512(%edx){1to8}, %ymm5, %ymm6 {%k7}

// CHECK: vcvtneps2bf16 %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x72,0xf5]
          vcvtneps2bf16 %xmm5, %xmm6

// CHECK: vcvtneps2bf16x  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x7e,0x0f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtneps2bf16x  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtneps2bf16   (%ecx){1to4}, %xmm6
// CHECK: encoding: [0x62,0xf2,0x7e,0x18,0x72,0x31]
          vcvtneps2bf16   (%ecx){1to4}, %xmm6

// CHECK: vcvtneps2bf16x  2032(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf2,0x7e,0x08,0x72,0x71,0x7f]
          vcvtneps2bf16x  2032(%ecx), %xmm6

// CHECK: vcvtneps2bf16   -512(%edx){1to4}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0x9f,0x72,0x72,0x80]
          vcvtneps2bf16   -512(%edx){1to4}, %xmm6 {%k7} {z}

// CHECK: vcvtneps2bf16 %ymm5, %xmm6
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x72,0xf5]
          vcvtneps2bf16 %ymm5, %xmm6

// CHECK: vcvtneps2bf16y  268435456(%esp,%esi,8), %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x7e,0x2f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtneps2bf16y  268435456(%esp,%esi,8), %xmm6 {%k7}

// CHECK: vcvtneps2bf16   (%ecx){1to8}, %xmm6
// CHECK: encoding: [0x62,0xf2,0x7e,0x38,0x72,0x31]
          vcvtneps2bf16   (%ecx){1to8}, %xmm6

// CHECK: vcvtneps2bf16y  4064(%ecx), %xmm6
// CHECK: encoding: [0x62,0xf2,0x7e,0x28,0x72,0x71,0x7f]
          vcvtneps2bf16y  4064(%ecx), %xmm6

// CHECK: vcvtneps2bf16   -512(%edx){1to8}, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0xbf,0x72,0x72,0x80]
          vcvtneps2bf16   -512(%edx){1to8}, %xmm6 {%k7} {z}

// CHECK: vdpbf16ps %ymm4, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf2,0x56,0x28,0x52,0xf4]
          vdpbf16ps %ymm4, %ymm5, %ymm6

// CHECK: vdpbf16ps   268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x56,0x2f,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdpbf16ps   268435456(%esp,%esi,8), %ymm5, %ymm6 {%k7}

// CHECK: vdpbf16ps   (%ecx){1to8}, %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf2,0x56,0x38,0x52,0x31]
          vdpbf16ps   (%ecx){1to8}, %ymm5, %ymm6

// CHECK: vdpbf16ps   4064(%ecx), %ymm5, %ymm6
// CHECK: encoding: [0x62,0xf2,0x56,0x28,0x52,0x71,0x7f]
          vdpbf16ps   4064(%ecx), %ymm5, %ymm6

// CHECK: vdpbf16ps   -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x56,0xbf,0x52,0x72,0x80]
          vdpbf16ps   -512(%edx){1to8}, %ymm5, %ymm6 {%k7} {z}

// CHECK: vdpbf16ps %xmm4, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf2,0x56,0x08,0x52,0xf4]
          vdpbf16ps %xmm4, %xmm5, %xmm6

// CHECK: vdpbf16ps   268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x56,0x0f,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdpbf16ps   268435456(%esp,%esi,8), %xmm5, %xmm6 {%k7}

// CHECK: vdpbf16ps   (%ecx){1to4}, %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf2,0x56,0x18,0x52,0x31]
          vdpbf16ps   (%ecx){1to4}, %xmm5, %xmm6

// CHECK: vdpbf16ps   2032(%ecx), %xmm5, %xmm6
// CHECK: encoding: [0x62,0xf2,0x56,0x08,0x52,0x71,0x7f]
          vdpbf16ps   2032(%ecx), %xmm5, %xmm6

// CHECK: vdpbf16ps   -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x56,0x9f,0x52,0x72,0x80]
          vdpbf16ps   -512(%edx){1to4}, %xmm5, %xmm6 {%k7} {z}

