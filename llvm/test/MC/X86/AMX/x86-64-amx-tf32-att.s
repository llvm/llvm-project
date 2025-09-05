// RUN: llvm-mc -triple x86_64-unknown-unknown --show-encoding < %s  | FileCheck %s

// CHECK:      tmmultf32ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x59,0x48,0xf5]
               tmmultf32ps %tmm4, %tmm5, %tmm6

// CHECK:      tmmultf32ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x71,0x48,0xda]
               tmmultf32ps %tmm1, %tmm2, %tmm3

// CHECK:      ttmmultf32ps %tmm4, %tmm5, %tmm6
// CHECK: encoding: [0xc4,0xe2,0x58,0x48,0xf5]
               ttmmultf32ps %tmm4, %tmm5, %tmm6

// CHECK:      ttmmultf32ps %tmm1, %tmm2, %tmm3
// CHECK: encoding: [0xc4,0xe2,0x70,0x48,0xda]
               ttmmultf32ps %tmm1, %tmm2, %tmm3
