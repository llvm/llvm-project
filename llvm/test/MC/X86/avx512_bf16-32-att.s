// RUN: llvm-mc -triple i686-unknown-unknown --show-encoding < %s | FileCheck %s

// CHECK: vcvtne2ps2bf16 %zmm4, %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0xf4]
          vcvtne2ps2bf16 %zmm4, %zmm5, %zmm6

// CHECK: vcvtne2ps2bf16 %zmm4, %zmm5, %zmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x57,0x4f,0x72,0xf4]
          vcvtne2ps2bf16 %zmm4, %zmm5, %zmm6 {%k7}

// CHECK: vcvtne2ps2bf16 %zmm4, %zmm5, %zmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x57,0xcf,0x72,0xf4]
          vcvtne2ps2bf16 %zmm4, %zmm5, %zmm6 {%k7} {z}

// CHECK: vcvtne2ps2bf16   (%ecx), %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0x31]
          vcvtne2ps2bf16   (%ecx), %zmm5, %zmm6

// CHECK: vcvtne2ps2bf16   291(%esp,%esi,8), %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0xb4,0xf4,0x23,0x01,0x00,0x00]
          vcvtne2ps2bf16   291(%esp,%esi,8), %zmm5, %zmm6

// CHECK: vcvtne2ps2bf16   268435456(%esp,%esi,8), %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtne2ps2bf16   268435456(%esp,%esi,8), %zmm5, %zmm6

// CHECK: vcvtne2ps2bf16   -64(%esp), %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0x74,0x24,0xff]
          vcvtne2ps2bf16   -64(%esp), %zmm5, %zmm6

// CHECK: vcvtne2ps2bf16   (%eax){1to16}, %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x58,0x72,0x30]
          vcvtne2ps2bf16   (%eax){1to16}, %zmm5, %zmm6

// CHECK: vcvtne2ps2bf16   8128(%edx), %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0x72,0x7f]
          vcvtne2ps2bf16   8128(%edx), %zmm5, %zmm6

// CHECK: vcvtne2ps2bf16   -8192(%edx), %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x48,0x72,0x72,0x80]
          vcvtne2ps2bf16   -8192(%edx), %zmm5, %zmm6

// CHECK: vcvtne2ps2bf16   508(%edx){1to16}, %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x58,0x72,0x72,0x7f]
          vcvtne2ps2bf16   508(%edx){1to16}, %zmm5, %zmm6

// CHECK: vcvtne2ps2bf16   -512(%edx){1to16}, %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x57,0x58,0x72,0x72,0x80]
          vcvtne2ps2bf16   -512(%edx){1to16}, %zmm5, %zmm6

// CHECK: vcvtneps2bf16 %zmm5, %ymm6
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x72,0xf5]
          vcvtneps2bf16 %zmm5, %ymm6

// CHECK: vcvtneps2bf16   268435456(%esp,%esi,8), %ymm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x7e,0x4f,0x72,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vcvtneps2bf16   268435456(%esp,%esi,8), %ymm6 {%k7}

// CHECK: vcvtneps2bf16   (%ecx){1to16}, %ymm6
// CHECK: encoding: [0x62,0xf2,0x7e,0x58,0x72,0x31]
          vcvtneps2bf16   (%ecx){1to16}, %ymm6

// CHECK: vcvtneps2bf16   8128(%ecx), %ymm6
// CHECK: encoding: [0x62,0xf2,0x7e,0x48,0x72,0x71,0x7f]
          vcvtneps2bf16   8128(%ecx), %ymm6

// CHECK: vcvtneps2bf16   -512(%edx){1to16}, %ymm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x7e,0xdf,0x72,0x72,0x80]
          vcvtneps2bf16   -512(%edx){1to16}, %ymm6 {%k7} {z}

// CHECK: vdpbf16ps %zmm4, %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x56,0x48,0x52,0xf4]
          vdpbf16ps %zmm4, %zmm5, %zmm6

// CHECK: vdpbf16ps   268435456(%esp,%esi,8), %zmm5, %zmm6 {%k7}
// CHECK: encoding: [0x62,0xf2,0x56,0x4f,0x52,0xb4,0xf4,0x00,0x00,0x00,0x10]
          vdpbf16ps   268435456(%esp,%esi,8), %zmm5, %zmm6 {%k7}

// CHECK: vdpbf16ps   (%ecx){1to16}, %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x56,0x58,0x52,0x31]
          vdpbf16ps   (%ecx){1to16}, %zmm5, %zmm6

// CHECK: vdpbf16ps   8128(%ecx), %zmm5, %zmm6
// CHECK: encoding: [0x62,0xf2,0x56,0x48,0x52,0x71,0x7f]
          vdpbf16ps   8128(%ecx), %zmm5, %zmm6

// CHECK: vdpbf16ps   -512(%edx){1to16}, %zmm5, %zmm6 {%k7} {z}
// CHECK: encoding: [0x62,0xf2,0x56,0xdf,0x52,0x72,0x80]
          vdpbf16ps   -512(%edx){1to16}, %zmm5, %zmm6 {%k7} {z}

