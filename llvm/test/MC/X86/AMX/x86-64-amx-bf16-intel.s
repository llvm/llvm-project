// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel -output-asm-variant=1 --show-encoding %s | FileCheck %s

// CHECK: tdpbf16ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5c,0xf5]
          tdpbf16ps tmm6, tmm5, tmm4

// CHECK: tdpbf16ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x72,0x5c,0xda]
          tdpbf16ps tmm3, tmm2, tmm1

// CHECK: tdpbf16ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5c,0xf5]
          tdpbf16ps tmm6, tmm5, tmm4

// CHECK: tdpbf16ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x72,0x5c,0xda]
          tdpbf16ps tmm3, tmm2, tmm1

// CHECK: tdpbf16ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5c,0xf5]
          tdpbf16ps tmm6, tmm5, tmm4

// CHECK: tdpbf16ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x72,0x5c,0xda]
          tdpbf16ps tmm3, tmm2, tmm1

// CHECK: tdpbf16ps tmm6, tmm5, tmm4
// CHECK: encoding: [0xc4,0xe2,0x5a,0x5c,0xf5]
          tdpbf16ps tmm6, tmm5, tmm4

// CHECK: tdpbf16ps tmm3, tmm2, tmm1
// CHECK: encoding: [0xc4,0xe2,0x72,0x5c,0xda]
          tdpbf16ps tmm3, tmm2, tmm1
