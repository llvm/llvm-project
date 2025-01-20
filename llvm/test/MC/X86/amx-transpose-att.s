// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding %s | FileCheck %s

// CHECK: t2rpntlvwz0     268435456(%rbp,%r14,8), %tmm4
// CHECK: encoding: [0xc4,0xa2,0x78,0x6e,0xa4,0xf5,0x00,0x00,0x00,0x10]
          t2rpntlvwz0 268435456(%rbp,%r14,8), %tmm4

// CHECK: t2rpntlvwz0     291(%r8,%rax,4), %tmm2
// CHECK: encoding: [0xc4,0xc2,0x78,0x6e,0x94,0x80,0x23,0x01,0x00,0x00]
          t2rpntlvwz0 291(%r8,%rax,4), %tmm2

// CHECK: t2rpntlvwz0     -32(,%rbp,2), %tmm2
// CHECK: encoding: [0xc4,0xe2,0x78,0x6e,0x14,0x6d,0xe0,0xff,0xff,0xff]
          t2rpntlvwz0 -32(,%rbp,2), %tmm2

// CHECK: t2rpntlvwz0t1     268435456(%rbp,%r14,8), %tmm4
// CHECK: encoding: [0xc4,0xa2,0x78,0x6f,0xa4,0xf5,0x00,0x00,0x00,0x10]
          t2rpntlvwz0t1 268435456(%rbp,%r14,8), %tmm5

// CHECK: t2rpntlvwz0t1     291(%r8,%rax,4), %tmm2
// CHECK: encoding: [0xc4,0xc2,0x78,0x6f,0x94,0x80,0x23,0x01,0x00,0x00]
          t2rpntlvwz0t1 291(%r8,%rax,4), %tmm2

// CHECK: t2rpntlvwz0t1     -32(,%rbp,2), %tmm2
// CHECK: encoding: [0xc4,0xe2,0x78,0x6f,0x14,0x6d,0xe0,0xff,0xff,0xff]
          t2rpntlvwz0t1 -32(,%rbp,2), %tmm2

// CHECK: t2rpntlvwz1     268435456(%rbp,%r14,8), %tmm4
// CHECK: encoding: [0xc4,0xa2,0x79,0x6e,0xa4,0xf5,0x00,0x00,0x00,0x10]
          t2rpntlvwz1 268435456(%rbp,%r14,8), %tmm5

// CHECK: t2rpntlvwz1     291(%r8,%rax,4), %tmm2
// CHECK: encoding: [0xc4,0xc2,0x79,0x6e,0x94,0x80,0x23,0x01,0x00,0x00]
          t2rpntlvwz1 291(%r8,%rax,4), %tmm2

// CHECK: t2rpntlvwz1     -32(,%rbp,2), %tmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x6e,0x14,0x6d,0xe0,0xff,0xff,0xff]
          t2rpntlvwz1 -32(,%rbp,2), %tmm2

// CHECK: t2rpntlvwz1t1     268435456(%rbp,%r14,8), %tmm2
// CHECK: encoding: [0xc4,0xa2,0x79,0x6f,0x94,0xf5,0x00,0x00,0x00,0x10]
          t2rpntlvwz1t1 268435456(%rbp,%r14,8), %tmm3

// CHECK: t2rpntlvwz1t1     291(%r8,%rax,4), %tmm2
// CHECK: encoding: [0xc4,0xc2,0x79,0x6f,0x94,0x80,0x23,0x01,0x00,0x00]
          t2rpntlvwz1t1 291(%r8,%rax,4), %tmm2

// CHECK: t2rpntlvwz1t1     -32(,%rbp,2), %tmm2
// CHECK: encoding: [0xc4,0xe2,0x79,0x6f,0x14,0x6d,0xe0,0xff,0xff,0xff]
          t2rpntlvwz1t1 -32(,%rbp,2), %tmm2

// CHECK: ttransposed     %tmm1, %tmm5
// CHECK: encoding: [0xc4,0xe2,0x7a,0x5f,0xe9]
          ttransposed %tmm1, %tmm5

// CHECK: ttransposed     %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x7a,0x5f,0xda]
          ttransposed %tmm2, %tmm3

// CHECK: ttdpbf16ps     %tmm1, %tmm2, %tmm5
// CHECK: encoding: [0xc4,0xe2,0x72,0x6c,0xea]
          ttdpbf16ps %tmm1, %tmm2, %tmm5

// CHECK: ttdpbf16ps     %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x6c,0xda]
          ttdpbf16ps %tmm1, %tmm2, %tmm3

// CHECK: ttdpfp16ps     %tmm3, %tmm4, %tmm5
// CHECK: encoding: [0xc4,0xe2,0x63,0x6c,0xec]
          ttdpfp16ps %tmm3, %tmm4, %tmm5

// CHECK: ttdpfp16ps     %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x73,0x6c,0xda]
          ttdpfp16ps %tmm1, %tmm2, %tmm3

// CHECK: ttcmmimfp16ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5b,0x6b,0xf5]
          ttcmmimfp16ps %tmm4, %tmm5, %tmm6

// CHECK: ttcmmimfp16ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x73,0x6b,0xda]
          ttcmmimfp16ps %tmm1, %tmm2, %tmm3

// CHECK: ttcmmrlfp16ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5a,0x6b,0xf5]
          ttcmmrlfp16ps %tmm4, %tmm5, %tmm6

// CHECK: ttcmmrlfp16ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x6b,0xda]
          ttcmmrlfp16ps %tmm1, %tmm2, %tmm3

// CHECK: tconjtcmmimfp16ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x58,0x6b,0xf5]
          tconjtcmmimfp16ps %tmm4, %tmm5, %tmm6

// CHECK: tconjtcmmimfp16ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x70,0x6b,0xda]
          tconjtcmmimfp16ps %tmm1, %tmm2, %tmm3

// CHECK: tconjtfp16 %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x79,0x6b,0xf5]
          tconjtfp16 %tmm5, %tmm6

// CHECK: tconjtfp16 %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x79,0x6b,0xda]
          tconjtfp16 %tmm2, %tmm3
