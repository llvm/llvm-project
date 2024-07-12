// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s | FileCheck %s

// CHECK: vp4dpwssd (%rax), %zmm20, %zmm17
// CHECK: encoding: [0x62,0xe2,0x5f,0x40,0x52,0x08]
vp4dpwssd (%rax), %zmm20, %zmm17
// CHECK: vp4dpwssd (%rax), %zmm8, %zmm3 {%k1}
// CHECK: encoding: [0x62,0xf2,0x3f,0x49,0x52,0x18]
vp4dpwssd (%rax), %zmm8, %zmm3 {k1}
// CHECK: vp4dpwssd (%rax), %zmm4, %zmm5 {%k1} {z}
// CHECK: encoding: [0x62,0xf2,0x5f,0xc9,0x52,0x28]
vp4dpwssd (%rax), %zmm4, %zmm5 {k1} {z}

// CHECK: vp4dpwssds (%rax), %zmm20, %zmm17
// CHECK: encoding: [0x62,0xe2,0x5f,0x40,0x53,0x08]
vp4dpwssds (%rax), %zmm20, %zmm17
// CHECK: vp4dpwssds (%rax), %zmm8, %zmm3 {%k1}
// CHECK: encoding: [0x62,0xf2,0x3f,0x49,0x53,0x18]
vp4dpwssds (%rax), %zmm8, %zmm3 {k1}
// CHECK: vp4dpwssds (%rax), %zmm4, %zmm5 {%k1} {z}
// CHECK: encoding: [0x62,0xf2,0x5f,0xc9,0x53,0x28]
vp4dpwssds (%rax), %zmm4, %zmm5 {k1} {z}
