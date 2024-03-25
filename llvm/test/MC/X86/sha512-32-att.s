// RUN: llvm-mc -triple i686 --show-encoding %s | FileCheck %s

// CHECK:      vsha512msg1 %xmm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x7f,0xcc,0xd3]
               vsha512msg1 %xmm3, %ymm2

// CHECK:      vsha512msg2 %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x7f,0xcd,0xd3]
               vsha512msg2 %ymm3, %ymm2

// CHECK:      vsha512rnds2 %xmm4, %ymm3, %ymm2
// CHECK: encoding: [0xc4,0xe2,0x67,0xcb,0xd4]
               vsha512rnds2 %xmm4, %ymm3, %ymm2
