// RUN: llvm-mc -triple x86_64 -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: tdpbf8ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe5,0x58,0xfd,0xf5]
          tdpbf8ps tmm6, tmm5, tmm4

// CHECK: tdpbf8ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe5,0x70,0xfd,0xda]
          tdpbf8ps tmm3, tmm2, tmm1

// CHECK: tdpbhf8ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe5,0x5b,0xfd,0xf5]
          tdpbhf8ps tmm6, tmm5, tmm4

// CHECK: tdpbhf8ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe5,0x73,0xfd,0xda]
          tdpbhf8ps tmm3, tmm2, tmm1

// CHECK: tdphbf8ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe5,0x5a,0xfd,0xf5]
          tdphbf8ps tmm6, tmm5, tmm4

// CHECK: tdphbf8ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe5,0x72,0xfd,0xda]
          tdphbf8ps tmm3, tmm2, tmm1

// CHECK: tdphf8ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe5,0x59,0xfd,0xf5]
          tdphf8ps tmm6, tmm5, tmm4

// CHECK: tdphf8ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe5,0x71,0xfd,0xda]
          tdphf8ps tmm3, tmm2, tmm1
