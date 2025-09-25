// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:      tmmultf32ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x59,0x48,0xf5]
               tmmultf32ps tmm6, tmm5, tmm4

// CHECK:      tmmultf32ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x48,0xda]
               tmmultf32ps tmm3, tmm2, tmm1

// CHECK:      ttmmultf32ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x58,0x48,0xf5]
               ttmmultf32ps tmm6, tmm5, tmm4

// CHECK:      ttmmultf32ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x70,0x48,0xda]
               ttmmultf32ps tmm3, tmm2, tmm1
