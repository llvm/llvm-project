// RUN: llvm-mc -triple x86_64-unknown-unknown -show-encoding %s | FileCheck %s
// some AMX instruction must use SIB.

// CHECK: tdpbf16ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5c,0xf5]
          tdpbf16ps %tmm4, %tmm5, %tmm6

// CHECK: tdpbf16ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x5c,0xda]
          tdpbf16ps %tmm1, %tmm2, %tmm3

// CHECK: tdpbf16ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5c,0xf5]
          tdpbf16ps %tmm4, %tmm5, %tmm6

// CHECK: tdpbf16ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x5c,0xda]
          tdpbf16ps %tmm1, %tmm2, %tmm3

// CHECK: tdpbf16ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5c,0xf5]
          tdpbf16ps %tmm4, %tmm5, %tmm6

// CHECK: tdpbf16ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x5c,0xda]
          tdpbf16ps %tmm1, %tmm2, %tmm3

// CHECK: tdpbf16ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5c,0xf5]
          tdpbf16ps %tmm4, %tmm5, %tmm6

// CHECK: tdpbf16ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x72,0x5c,0xda]
          tdpbf16ps %tmm1, %tmm2, %tmm3
