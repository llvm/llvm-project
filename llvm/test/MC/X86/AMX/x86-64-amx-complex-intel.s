// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK:      tcmmimfp16ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x59,0x6c,0xf5]
               tcmmimfp16ps tmm6, tmm5, tmm4

// CHECK:      tcmmimfp16ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x71,0x6c,0xda]
               tcmmimfp16ps tmm3, tmm2, tmm1

// CHECK:      tcmmrlfp16ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x58,0x6c,0xf5]
               tcmmrlfp16ps tmm6, tmm5, tmm4

// CHECK:      tcmmrlfp16ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x70,0x6c,0xda]
               tcmmrlfp16ps tmm3, tmm2, tmm1
