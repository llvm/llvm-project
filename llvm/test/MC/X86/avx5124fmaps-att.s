// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s > %t 2> %t.err
// RUN: FileCheck < %t %s
// RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s

// CHECK: v4fmaddps (%rax), %zmm20, %zmm17
// CHECK: encoding: [0x62,0xe2,0x5f,0x40,0x9a,0x08]
v4fmaddps (%rax), %zmm20, %zmm17
// CHECK: v4fmaddps (%rax), %zmm8, %zmm3 {%k1}
// CHECK: encoding: [0x62,0xf2,0x3f,0x49,0x9a,0x18]
v4fmaddps (%rax), %zmm8, %zmm3 {k1}
// CHECK: v4fmaddps (%rax), %zmm4, %zmm5 {%k1} {z}
// CHECK: encoding: [0x62,0xf2,0x5f,0xc9,0x9a,0x28]
v4fmaddps (%rax), %zmm4, %zmm5 {k1} {z}

// CHECK: v4fmaddss (%rax), %xmm20, %xmm17
// CHECK: encoding: [0x62,0xe2,0x5f,0x00,0x9b,0x08]
v4fmaddss (%rax), %xmm20, %xmm17
// CHECK: v4fmaddss (%rax), %xmm8, %xmm3 {%k1}
// CHECK: encoding: [0x62,0xf2,0x3f,0x09,0x9b,0x18]
v4fmaddss (%rax), %xmm8, %xmm3 {k1}
// CHECK: v4fmaddss (%rax), %xmm4, %xmm5 {%k1} {z}
// CHECK: encoding: [0x62,0xf2,0x5f,0x89,0x9b,0x28]
v4fmaddss (%rax), %xmm4, %xmm5 {k1} {z}

// CHECK: v4fnmaddps (%rax), %zmm20, %zmm17
// CHECK: encoding: [0x62,0xe2,0x5f,0x40,0xaa,0x08]
v4fnmaddps (%rax), %zmm20, %zmm17
// CHECK: v4fnmaddps (%rax), %zmm8, %zmm3 {%k1}
// CHECK: encoding: [0x62,0xf2,0x3f,0x49,0xaa,0x18]
v4fnmaddps (%rax), %zmm8, %zmm3 {k1}
// CHECK: v4fnmaddps (%rax), %zmm4, %zmm5 {%k1} {z}
// CHECK: encoding: [0x62,0xf2,0x5f,0xc9,0xaa,0x28]
v4fnmaddps (%rax), %zmm4, %zmm5 {k1} {z}

// CHECK: v4fnmaddss (%rax), %xmm20, %xmm17
// CHECK: encoding: [0x62,0xe2,0x5f,0x00,0xab,0x08]
v4fnmaddss (%rax), %xmm20, %xmm17
// CHECK: v4fnmaddss (%rax), %xmm8, %xmm3 {%k1}
// CHECK: encoding: [0x62,0xf2,0x3f,0x09,0xab,0x18]
v4fnmaddss (%rax), %xmm8, %xmm3 {k1}
// CHECK: v4fnmaddss (%rax), %xmm4, %xmm5 {%k1} {z}
// CHECK: encoding: [0x62,0xf2,0x5f,0x89,0xab,0x28]
v4fnmaddss (%rax), %xmm4, %xmm5 {k1} {z}


// CHECK-STDERR: warning: source register 'zmm21' implicitly denotes 'zmm20' to 'zmm23' source group
// CHECK: v4fmaddps (%rax), %zmm21, %zmm17
// CHECK: encoding: [0x62,0xe2,0x57,0x40,0x9a,0x08]
v4fmaddps (%rax), %zmm21, %zmm17
// CHECK-STDERR: warning: source register 'zmm10' implicitly denotes 'zmm8' to 'zmm11' source group
// CHECK: v4fmaddps (%rax), %zmm10, %zmm3 {%k1}
// CHECK: encoding: [0x62,0xf2,0x2f,0x49,0x9a,0x18]
v4fmaddps (%rax), %zmm10, %zmm3 {k1}
// CHECK-STDERR: warning: source register 'zmm7' implicitly denotes 'zmm4' to 'zmm7' source group
// CHECK: v4fmaddps (%rax), %zmm7, %zmm5 {%k1} {z}
// CHECK: encoding: [0x62,0xf2,0x47,0xc9,0x9a,0x28]
v4fmaddps (%rax), %zmm7, %zmm5 {k1} {z}

// CHECK-STDERR: warning: source register 'xmm21' implicitly denotes 'xmm20' to 'xmm23' source group
// CHECK: v4fmaddss (%rax), %xmm21, %xmm17
// CHECK: encoding: [0x62,0xe2,0x57,0x00,0x9b,0x08]
v4fmaddss (%rax), %xmm21, %xmm17
// CHECK-STDERR: warning: source register 'xmm10' implicitly denotes 'xmm8' to 'xmm11' source group
// CHECK: v4fmaddss (%rax), %xmm10, %xmm3 {%k1}
// CHECK: encoding: [0x62,0xf2,0x2f,0x09,0x9b,0x18]
v4fmaddss (%rax), %xmm10, %xmm3 {k1}
// CHECK-STDERR: warning: source register 'xmm7' implicitly denotes 'xmm4' to 'xmm7' source group
// CHECK: v4fmaddss (%rax), %xmm7, %xmm5 {%k1} {z}
// CHECK: encoding: [0x62,0xf2,0x47,0x89,0x9b,0x28]
v4fmaddss (%rax), %xmm7, %xmm5 {k1} {z}
