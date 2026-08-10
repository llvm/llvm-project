// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s  | FileCheck %s

// CHECK:      tcmmimfp16ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x59,0x6c,0xf5]
               tcmmimfp16ps %tmm4, %tmm5, %tmm6

// CHECK:      tcmmimfp16ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x71,0x6c,0xda]
               tcmmimfp16ps %tmm1, %tmm2, %tmm3

// CHECK:      tcmmrlfp16ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x58,0x6c,0xf5]
               tcmmrlfp16ps %tmm4, %tmm5, %tmm6

// CHECK:      tcmmrlfp16ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x70,0x6c,0xda]
               tcmmrlfp16ps %tmm1, %tmm2, %tmm3
