// RUN: llvm-mc -triple x86_64 --show-encoding %s | FileCheck %s

// CHECK: tdpbf8ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe5,0x58,0xfd,0xf5]
          tdpbf8ps %tmm4, %tmm5, %tmm6

// CHECK: tdpbf8ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe5,0x70,0xfd,0xda]
          tdpbf8ps %tmm1, %tmm2, %tmm3

// CHECK: tdpbhf8ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe5,0x5b,0xfd,0xf5]
          tdpbhf8ps %tmm4, %tmm5, %tmm6

// CHECK: tdpbhf8ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe5,0x73,0xfd,0xda]
          tdpbhf8ps %tmm1, %tmm2, %tmm3

// CHECK: tdphbf8ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe5,0x5a,0xfd,0xf5]
          tdphbf8ps %tmm4, %tmm5, %tmm6

// CHECK: tdphbf8ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe5,0x72,0xfd,0xda]
          tdphbf8ps %tmm1, %tmm2, %tmm3

// CHECK: tdphf8ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe5,0x59,0xfd,0xf5]
          tdphf8ps %tmm4, %tmm5, %tmm6

// CHECK: tdphf8ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe5,0x71,0xfd,0xda]
          tdphf8ps %tmm1, %tmm2, %tmm3
